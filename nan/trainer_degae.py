import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from configs.local_setting import OUT_DIR, LOG_DIR
from nan.criterion import NANLoss
from nan.dataloaders import dataset_dict
from nan.dataloaders.basic_dataset import de_linearize, Mode
from nan.dataloaders.create_training_dataset import create_training_dataset
from nan.dataloaders.data_utils import cycle
from nan.losses import l2_loss, VGG
from nan.model import NANScheme
from nan.render_image import render_single_image
from nan.render_ray import RayRender
from nan.sample_ray import RaySampler
from nan.utils.eval_utils import mse2psnr, img2psnr
from nan.utils.general_utils import img_HWC2CHW
from nan.utils.io_utils import print_link, colorize
# from pytorch_msssim import ms_ssim
from nan.ssim_l1_loss import MS_SSIM_L1_LOSS
from runpy import run_path

from nan.content_loss import reconstruction_loss
from degae.model import DegAE
from degae.esrgan.discriminator import DiscriminatorUNet
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import torch.nn.functional as F
from pathlib import Path

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.local_rank}")

        # exp_out_dir will contain .pth weights and .yml config files for the experiment
        self.exp_out_dir = OUT_DIR / args.expname
        self.exp_out_dir.mkdir(exist_ok=True, parents=True)
        print_link(self.exp_out_dir, 'outputs will be saved to')

        # save the args and config files
        self.save_ymls(args, sys.argv[1:], self.exp_out_dir)

        # create training dataset
        args.eval_gain = [0,1,20,16]
        self.train_dataset, self.train_sampler = create_training_dataset(args)
        # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
        # please use distributed parallel on multiple GPUs to train multiple target views per batch
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=args.batch_size,
                                                        worker_init_fn=lambda _: np.random.seed(),
                                                        num_workers=args.workers,
                                                        pin_memory=True,
                                                        sampler=self.train_sampler,
                                                        shuffle=True if self.train_sampler is None else False)

        # create validation dataset
        self.val_dataset = dataset_dict[args.eval_dataset](args, Mode.validation, scenes=args.eval_scenes)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1)
        self.val_loader_iterator = iter(cycle(self.val_loader))

        self.model = DegAE(args, train_scratch=True)


        # For perceptual Loss
        self.vgg_loss = VGG().to(self.device)
        self.vgg_mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.vgg_std =  torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

        # For adversarial Loss
        if self.args.lambda_adv > 0:
            self.discriminator = DiscriminatorUNet(in_channels=3, out_channels=1, channels=64).to(self.device)
            self.adv_loss = nn.BCEWithLogitsLoss()
            params_list = [{'params': self.discriminator.parameters(), 'lr': self.args.lrate_feature}]
            self.d_optimizer = torch.optim.Adam(params_list)
            self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer,
                                                        step_size=self.args.lrate_decay_steps,
                                                        gamma=self.args.lrate_decay_factor)


        if args.ckpt_path != None:
            ckpts = torch.load(args.ckpt_path)
            self.model.load_state_dict(ckpts['model'])

            if args.lambda_adv > 0:
                assert args.discrim_ckpt_path != None
                discrim_ckpts = torch.load(args.discrim_ckpt_path)
                self.discriminator.load_state_dict(discrim_ckpts['model'])
                
        # tb_dir will contain tensorboard files and later evaluation results
        tb_dir = LOG_DIR / args.expname
        if args.local_rank == 0:
            self.writer = SummaryWriter(str(tb_dir))
            print_link(tb_dir, 'saving tensorboard files to')
        # dictionary to store scalars to log in tb
        self.scalars_to_log = {}


    @staticmethod
    def save_ymls(args, additional_args, out_folder):
        with open(out_folder / 'args.yml', 'w') as f:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                f.write(f'{arg} : {attr}\n')

        if args.config is not None:
            with open(str(args.config)) as f:
                contents = yaml.safe_load(f)
        else:
            contents = {}
        for arg in filter(lambda s: s[:2] == '--', additional_args):
            val = vars(args)[arg[2:]]
            if isinstance(val, Path):
                val = str(val)
            contents[arg[2:]] = val

        with open(out_folder / 'config.yml', 'w') as f:
            yaml.safe_dump(contents, f, default_flow_style=None)

    def train(self):
        global_step = 0 #self.model.start_step + 1
        epoch = 0  # epoch is not consistent when loading ckpt, it affects train_sampler when distributed and prints

        while global_step < self.args.n_iters + 1:
            np.random.seed()
            for train_data in self.train_loader:
                time0 = time.time()
                if self.args.distributed:
                    self.train_sampler.set_epoch(epoch)

                self.model.encoder.train()
                self.model.degrep_extractor.degrep_conv.train()
                self.model.degrep_extractor.degrep_fc.train()
                self.model.decoder.train()

                # core optimization loop
                self.training_loop(train_data, global_step)

                self.model.encoder.eval()
                self.model.degrep_extractor.degrep_conv.eval()
                self.model.degrep_extractor.degrep_fc.eval()
                self.model.decoder.eval()

                # Logging and saving
                with torch.no_grad():
                    self.logging(train_data, global_step, epoch)

                global_step += 1
            epoch += 1
        return self.last_weights_path

    def training_loop(self, train_data, global_step):
        """

        :param train_data: dict {camera: (B, 34),
                                 src_rgbs_clean: (B, N, H, W, 3),
                                 src_rgbs: (B, N, H, W, 3),
                                 src_cameras: (B, N, 34),
                                 depth_range: (1, 2),
                                 sigma_estimate: (B, N, H, W, 3),
                                 white_level: (1, 1),
                                 rgb_clean: (B, H, W, 3), rgb: (B, H, W, 3),
                                 gt_depth: ,
                                 rgb_path: list(B)}
        :return:
        """
        for k in train_data.keys():
            train_data[k] = train_data[k].to(self.device)

        reconst_signal = self.model(train_data)
        self.model.optimizer.zero_grad()

        delin_pred = de_linearize(reconst_signal, train_data['white_level'])
        delin_tar = de_linearize(train_data['target_rgb'], train_data['white_level'])
        loss = 0
        
        # loss1
        content_loss = reconstruction_loss(reconst_signal, train_data['target_rgb'], self.device) * self.args.lambda_content
        loss += content_loss 

        
        # loss2
        if self.args.lambda_perceptual > 0:
            delin_norm_pred = (delin_pred.clamp(0,1) - self.vgg_mean.reshape(1,-1,1,1)) / self.vgg_std.reshape(1,-1,1,1)
            delin_norm_tar = (delin_tar.clamp(0,1) - self.vgg_mean.reshape(1,-1,1,1)) / self.vgg_std.reshape(1,-1,1,1)
            vgg_feat_pred = self.vgg_loss.vgg(delin_norm_pred)
            vgg_feat_tar = self.vgg_loss.vgg(delin_norm_tar)
            perceptual_loss = F.mse_loss(vgg_feat_pred, vgg_feat_tar, reduction='mean')  * self.args.lambda_perceptual
            loss += perceptual_loss
            self.scalars_to_log['train/perceptual_loss'] = perceptual_loss

        # loss3        
        if not self.args.skip_condition:        
            noise_vec_tar = self.model.degrep_extractor(train_data['target_rgb'], train_data['white_level'])
            noise_vec_pred = self.model.degrep_extractor(reconst_signal, train_data['white_level'])
            embed_loss = F.mse_loss(noise_vec_tar, noise_vec_pred) * self.args.lambda_embed
            loss += embed_loss
            self.scalars_to_log['train/embed_loss'] = embed_loss

        # loss4
        real_label = torch.full([reconst_signal.shape[0], 1, reconst_signal.shape[-2], reconst_signal.shape[-1]], 1.0, dtype=torch.float, device=reconst_signal.device)
        fake_label = torch.full([reconst_signal.shape[0], 1, reconst_signal.shape[-2], reconst_signal.shape[-1]], 0.0, dtype=torch.float, device=reconst_signal.device)
        if self.args.lambda_adv > 0:

            for d_parameters in self.discriminator.parameters():
                d_parameters.requires_grad = False

            adversarial_loss = self.adv_loss(self.discriminator(delin_pred), real_label)
            adversarial_loss = torch.mean(adversarial_loss) * self.args.lambda_adv
            loss += adversarial_loss
            self.scalars_to_log['train/adversarial_loss'] = adversarial_loss

        loss.backward()
        self.model.optimizer.step()
        self.model.scheduler.step()

        l2_loss = torch.mean((delin_pred.clamp(0,1) - delin_tar.clamp(0,1)) ** 2)
        self.scalars_to_log['train/content_loss'] = content_loss
        self.scalars_to_log['train/l2_loss'] = l2_loss
        self.scalars_to_log['train/psnr'] = mse2psnr(l2_loss.detach().cpu())



        if self.args.lambda_adv > 0:
            self.discriminator.zero_grad(set_to_none=True)
            for d_parameters in self.discriminator.parameters():
                d_parameters.requires_grad = True

            self.d_optimizer.zero_grad()
            gt_output = self.discriminator(delin_tar)
            fake_output = self.discriminator(delin_pred.detach().clone())

            d_loss_gt = self.adv_loss(gt_output, real_label)
            d_loss_fake = self.adv_loss(fake_output, fake_label)

            d_loss = torch.mean(d_loss_fake) + torch.mean(d_loss_gt)
            d_loss.backward()
            self.d_optimizer.step()
            self.d_scheduler.step()
            self.scalars_to_log['train/d_loss'] = d_loss

        # print(round(loss.item(),4), round(perceptual_loss.item(),4), round(content_loss.item(),4), round(embed_loss.item(),4), mse2psnr(l2_loss.detach().cpu())) # 

    def logging(self, train_data, global_step, epoch, max_keep=3):
        if self.args.local_rank == 0:
            # log iteration values
            if global_step % self.args.i_tb == 0 or global_step < 10:
                # write mse and psnr stats
                logstr = f"{self.args.expname} Epoch: {epoch}  step: {global_step} "
                for k in self.scalars_to_log.keys():
                    logstr += f" {k}: {self.scalars_to_log[k]:.6f}"
                    self.writer.add_scalar(k, self.scalars_to_log[k], global_step)

                if global_step % self.args.i_print == 0:
                    print(logstr)

            # save weights
            if global_step % self.args.i_weights == 0:
                print(f"Saving checkpoints at {global_step} to {self.exp_out_dir}...")
                self.last_weights_path = self.exp_out_dir / f"model_{global_step:06d}.pth"
                self.model.save_model(self.last_weights_path)
                if self.args.lambda_adv > 0:
                    last_weights_path_discrim = self.exp_out_dir / f"discrim_{global_step:06d}.pth"
                    self.discriminator.save_model(last_weights_path_discrim, self.d_optimizer, self.d_scheduler)

                model_files = sorted(self.exp_out_dir.glob("*.pth"), key=os.path.getctime)
                model_files = [f for f in model_files if 'model' in str(f)]
                rm_files = model_files[0:max(0, len(model_files) - max_keep)]
                for f in rm_files:
                    f.unlink()

                if self.args.lambda_adv > 0:
                    discrim_files = sorted(self.exp_out_dir.glob("*.pth"), key=os.path.getctime)
                    discrim_files = [f for f in discrim_files if 'discrim' in str(f)]
                    rm_files = discrim_files[0:max(0, len(discrim_files) - max_keep)]
                    for f in rm_files:
                        f.unlink()

            # log images of training and validation
            if global_step % self.args.i_img == 0: #or global_step == self.model.start_step + 1:
                self.log_images(train_data, global_step)


    def log_view_to_tb(self, global_step, batch_data, prefix='', idx=0, visualize=False):
        reconst_signal = self.model(batch_data)

        delin_pred = de_linearize(reconst_signal, batch_data['white_level']).clamp(0,1)
        delin_tar = de_linearize(batch_data['target_rgb'], batch_data['white_level']).clamp(0,1)
        delin_ref = de_linearize(batch_data['ref_rgb'], batch_data['white_level']).clamp(0,1)
        delin_noisy = de_linearize(batch_data['noisy_rgb'], batch_data['white_level']).clamp(0,1)

        if 'eval_gain' in batch_data.keys():
            eval_gain = batch_data['eval_gain']
        else:
            eval_gain = 0
        
        if visualize:
            self.writer.add_image(prefix + f'reconst_rgb_gain{eval_gain}_{idx}', delin_pred[0].detach().cpu(), global_step)
            if global_step < 5 or 'train' in prefix:
                self.writer.add_image(prefix + f'target_rgb_gain_{idx}', delin_tar[0].detach().cpu(), global_step)
                self.writer.add_image(prefix + f'input_rgb_gain{eval_gain}_{idx}', delin_noisy[0].detach().cpu(), global_step)


        l2_loss = F.mse_loss(delin_tar, delin_pred, reduction='mean')
        psnr = mse2psnr(l2_loss.detach().cpu())        
        return psnr

    def log_images(self, train_data, global_step):
        print('Logging a random validation view...')
        val_result = {}
        psnr_results = {}
        val_interval = 1

        for val_idx in range(len(self.val_dataset)):
            curr_idx = val_idx % len(self.val_dataset.render_rgb_files)
            if curr_idx % 15 == 0:
                visualize = True
            elif curr_idx % val_interval == 0 :
                visualize = False
            else:
                continue

            val_data = self.val_dataset[val_idx]
            val_data = {k : val_data[k][None].to(self.device) if isinstance(val_data[k], torch.Tensor) else val_data[k] for k in val_data.keys()}
            psnr = self.log_view_to_tb(global_step, val_data, prefix='val/', idx=curr_idx , visualize=visualize)
            eval_gain = val_data['eval_gain']
            if eval_gain in psnr_results.keys():
                psnr_results[eval_gain].append(psnr)
            else:
                psnr_results[eval_gain] = [psnr]

            torch.cuda.empty_cache()
            # print("Val # img", val_idx)
        for k in psnr_results.keys():
            self.writer.add_scalar('val/' + f'psnr_gain{k}', np.mean(psnr_results[k]), global_step)
            
        self.log_view_to_tb(global_step, train_data, prefix=f'train/', visualize=True)



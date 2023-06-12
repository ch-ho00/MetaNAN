import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
        args.eval_gain = [20,16,8,4,2,1]
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

        self.model = DegAE(args)

        self.vgg_loss = VGG().to(self.device)
        self.vgg_mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.vgg_std =  torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

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
                self.model.degrep_extractor.train()
                self.model.decoder.train()

                # core optimization loop
                self.training_loop(train_data, global_step)

                self.model.encoder.eval()
                self.model.degrep_extractor.eval()
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
        perceptual_loss = torch.zeros_like(content_loss)
        delin_norm_pred = (delin_pred.clamp(0,1) - self.vgg_mean.reshape(1,-1,1,1)) / self.vgg_std.reshape(1,-1,1,1)
        delin_norm_tar = (delin_tar.clamp(0,1) - self.vgg_mean.reshape(1,-1,1,1)) / self.vgg_std.reshape(1,-1,1,1)
        vgg_feat_pred = self.vgg_loss.vgg(delin_norm_pred)
        vgg_feat_tar = self.vgg_loss.vgg(delin_norm_tar)
        perceptual_loss = F.mse_loss(vgg_feat_pred, vgg_feat_tar, reduction='mean')  * self.args.lambda_perceptual
        loss += perceptual_loss

        # loss3        
        embed_loss = torch.zeros_like(content_loss)
        if self.args.condition_decode:        
            noise_vec_tar = self.model.degrep_extractor(train_data['target_rgb'], train_data['white_level'])
            noise_vec_pred = self.model.degrep_extractor(reconst_signal, train_data['white_level'])
            embed_loss = F.mse_loss(noise_vec_tar, noise_vec_pred) * self.args.lambda_embed
            loss += embed_loss

        loss.backward()
        self.model.optimizer.step()
        self.model.scheduler.step()

        l2_loss = torch.mean((delin_pred.clamp(0,1) - delin_tar.clamp(0,1)) ** 2)

        self.scalars_to_log['train/content_loss'] = content_loss
        self.scalars_to_log['train/embed_loss'] = embed_loss
        self.scalars_to_log['train/perceptual_loss'] = perceptual_loss
        self.scalars_to_log['train/l2_loss'] = l2_loss
        self.scalars_to_log['train/psnr'] = mse2psnr(l2_loss.detach().cpu())

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
                files = sorted(self.exp_out_dir.glob("*.pth"), key=os.path.getctime)
                rm_files = files[0:max(0, len(files) - max_keep)]
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
        delin_clean = de_linearize(batch_data['clean_rgb'], batch_data['white_level']).clamp(0,1)
        delin_noisy = de_linearize(batch_data['noisy_rgb'], batch_data['white_level']).clamp(0,1)

        if 'eval_gain' in batch_data.keys():
            eval_gain = batch_data['eval_gain']
        else:
            eval_gain = 0
        
        if visualize:
            self.writer.add_image(prefix + f'target_rgb_gain{eval_gain}_{idx}', delin_tar[0].detach().cpu(), global_step)
            self.writer.add_image(prefix + f'reconst_rgb_gain{eval_gain}_{idx}', delin_pred[0].detach().cpu(), global_step)
            self.writer.add_image(prefix + f'input_rgb_gain{eval_gain}_{idx}', delin_noisy[0].detach().cpu(), global_step)


        l2_loss = F.mse_loss(delin_tar, delin_pred, reduction='mean')
        psnr = mse2psnr(l2_loss.detach().cpu())        
        return psnr

    def log_images(self, train_data, global_step):
        print('Logging a random validation view...')
        val_result = {}
        for val_idx in range(len(self.val_dataset)):
            curr_idx = val_idx % len(self.val_dataset.render_rgb_files)
            if curr_idx > 5 and curr_idx < len(self.val_dataset.render_rgb_files) - 5:
                continue            
            val_data = self.val_dataset[val_idx]
            val_data = {k : val_data[k][None].to(self.device) if isinstance(val_data[k], torch.Tensor) else val_data[k] for k in val_data.keys()}
            val_psnr = self.log_view_to_tb(global_step, val_data, prefix='val/', idx=curr_idx , visualize=curr_idx == 0 or curr_idx == len(self.val_dataset.render_rgb_files)-1)
            eval_gain = val_data['eval_gain']
            if eval_gain not in val_result.keys():
                val_result[eval_gain] = [val_psnr]
            else:
                val_result[eval_gain] += [val_psnr]
            torch.cuda.empty_cache()

        for gain_level in val_result.keys():
            self.writer.add_scalar(f'val/psnr_gain{gain_level}', np.mean(val_result[gain_level]), global_step)
            
        self.log_view_to_tb(global_step, train_data, prefix=f'train/', visualize=True)



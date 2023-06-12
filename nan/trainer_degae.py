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
from degae.uformer.model import Uformer
from degae.srgan.vgg import DegFeatureExtractor
from degae.decoder import DegAE_decoder
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import torch.nn.functional as F

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
        args.eval_gain = [20,16,8]
        self.train_dataset, self.train_sampler = create_training_dataset(args)
        # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
        # please use distributed parallel on multiple GPUs to train multiple target views per batch
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=2, # args.batch_size
                                                        worker_init_fn=lambda _: np.random.seed(),
                                                        num_workers=args.workers,
                                                        pin_memory=True,
                                                        sampler=self.train_sampler,
                                                        shuffle=True if self.train_sampler is None else False)

        # create validation dataset
        self.val_dataset = dataset_dict[args.eval_dataset](args, Mode.validation, scenes=args.eval_scenes)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1)
        self.val_loader_iterator = iter(cycle(self.val_loader))

        # Create NAN scheme
        ## 
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
        img_wh = [512, 512] #[1024, 768]
        self.encoder = Uformer(img_wh=img_wh, embed_dim=16, depths=depths,
                    win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False).to(self.device)

        self.degrep_extractor = DegFeatureExtractor(args.degrep_ckpt).to(self.device)
        self.decoder = DegAE_decoder().to(self.device)
        self.vgg_loss = VGG().to(self.device)
        self.optimizer, self.scheduler = self.create_optimizer()
        self.vgg_mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.vgg_std =  torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

        # tb_dir will contain tensorboard files and later evaluation results
        tb_dir = LOG_DIR / args.expname
        if args.local_rank == 0:
            self.writer = SummaryWriter(str(tb_dir))
            print_link(tb_dir, 'saving tensorboard files to')
        # dictionary to store scalars to log in tb
        self.scalars_to_log = {}


    def create_optimizer(self):
        params_list = [{'params': self.encoder.parameters(), 'lr': self.args.lrate_feature},
                       {'params': self.degrep_extractor.degrep_conv.parameters(),  'lr': self.args.lrate_feature},
                       {'params': self.degrep_extractor.degrep_fc.parameters(),  'lr': self.args.lrate_feature},
                       {'params': self.decoder.parameters(),  'lr': self.args.lrate_feature},                       
                       ]
                    #    {'params': self.degrep_extractor.vgg.parameters(),  'lr': self.args.lrate_feature},
        optimizer = torch.optim.Adam(params_list)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.args.lrate_decay_steps,
                                                    gamma=self.args.lrate_decay_factor)

        return optimizer, scheduler


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

                self.encoder.train()
                self.degrep_extractor.train()
                self.decoder.train()

                # core optimization loop
                self.training_loop(train_data, global_step)

                self.encoder.eval()
                self.degrep_extractor.eval()
                self.decoder.eval()

                # Logging and saving
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
        img_embed = self.encoder(train_data['noisy_rgb'])
        noise_vec_ref = None
        if self.args.condition_decode:        
            noise_vec_ref = self.degrep_extractor(train_data['ref_rgb'], train_data['white_level'][0])
        
        reconst_signal = self.decoder(img_embed, noise_vec_ref)
        self.optimizer.zero_grad()
        loss = 0
        loss_dict = {}
        
        # loss1
        content_loss = reconstruction_loss(reconst_signal, train_data['target_rgb'], self.device) * 0.1
        loss += content_loss 

        
        # loss2
        delin_pred = de_linearize(reconst_signal, train_data['white_level'][0])
        delin_tar = de_linearize(train_data['target_rgb'], train_data['white_level'][0])


        perceptual_loss = torch.zeros_like(content_loss)
        delin_norm_pred = (delin_pred.clamp(0,1) - self.vgg_mean.reshape(1,-1,1,1)) / self.vgg_std.reshape(1,-1,1,1)
        delin_norm_tar = (delin_tar.clamp(0,1) - self.vgg_mean.reshape(1,-1,1,1)) / self.vgg_std.reshape(1,-1,1,1)
        vgg_feat_pred = self.vgg_loss.vgg(delin_norm_pred)
        vgg_feat_tar = self.vgg_loss.vgg(delin_norm_tar)
        perceptual_loss = F.mse_loss(vgg_feat_pred, vgg_feat_tar, reduction='mean')  * 1e-2
        loss += perceptual_loss
        # ssim_loss = 1 - ssim( , data_range=1, size_average=True) # return a scalar

        # loss3        
        embed_loss = torch.zeros_like(content_loss)
        if self.args.condition_decode:        
            noise_vec_tar = self.degrep_extractor(train_data['target_rgb'], train_data['white_level'][0])
            noise_vec_pred = self.degrep_extractor(reconst_signal, train_data['white_level'][0])
            embed_loss = F.mse_loss(noise_vec_tar, noise_vec_pred)
            loss += embed_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()

        l2_loss = torch.mean((delin_pred.clamp(0,1) - delin_tar.clamp(0,1)) ** 2)

        self.scalars_to_log['train/content_loss'] = content_loss
        self.scalars_to_log['train/embed_loss'] = embed_loss
        self.scalars_to_log['train/perceptual_loss'] = perceptual_loss
        self.scalars_to_log['train/l2_loss'] = l2_loss
        self.scalars_to_log['train/psnr'] = mse2psnr(l2_loss.detach().cpu())

        print(round(loss.item(),4), round(perceptual_loss.item(),4), round(content_loss.item(),4), round(embed_loss.item(),4), mse2psnr(l2_loss.detach().cpu())) # 

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
            # if global_step % self.args.i_weights == 0:
            #     print(f"Saving checkpoints at {global_step} to {self.exp_out_dir}...")
            #     self.last_weights_path = self.exp_out_dir / f"model_{global_step:06d}.pth"
            #     self.model.save_model(self.last_weights_path)
            #     files = sorted(self.exp_out_dir.glob("*.pth"), key=os.path.getctime)
            #     rm_files = files[0:max(0, len(files) - max_keep)]
            #     for f in rm_files:
            #         f.unlink()

            # log images of training and validation
            if global_step % self.args.i_img == 0: #or global_step == self.model.start_step + 1:
                self.log_images(train_data, global_step)


    def log_view_to_tb(self, global_step, batch_data, prefix='', fn=''):
        img_embed = self.encoder(batch_data['noisy_rgb'])
        noise_vec = None
        if self.args.condition_decode:        
            noise_vec = self.degrep_extractor(batch_data['ref_rgb'], batch_data['white_level'][0])
        
        reconst_signal = self.decoder(img_embed, noise_vec)
        delin_pred = de_linearize(reconst_signal, batch_data['white_level'][0]).clamp(0,1)
        delin_tar = de_linearize(batch_data['target_rgb'], batch_data['white_level'][0]).clamp(0,1)
        delin_ref = de_linearize(batch_data['ref_rgb'], batch_data['white_level'][0]).clamp(0,1)
        delin_clean = de_linearize(batch_data['clean_rgb'], batch_data['white_level'][0]).clamp(0,1)
        delin_noisy = de_linearize(batch_data['noisy_rgb'], batch_data['white_level'][0]).clamp(0,1)

        # import pdb; pdb.set_trace()
        # self.writer.add_image(prefix + "reconst_" + fn , delin_pred[0], global_step)
        # self.writer.add_image(prefix + "gt_" + fn , delin_tar[0], global_step)
        plt.imsave(f'./degrad_ae_128/reconst_{fn}_{global_step}.png', delin_pred[0].detach().cpu().permute(1,2,0).numpy())
        plt.imsave(f'./degrad_ae_128/tar_{fn}_{global_step}.png', delin_tar[0].detach().cpu().permute(1,2,0).numpy())
        plt.imsave(f'./degrad_ae_128/ref_{fn}_{global_step}.png', delin_ref[0].detach().cpu().permute(1,2,0).numpy())
        plt.imsave(f'./degrad_ae_128/clean_{fn}_{global_step}.png', delin_clean[0].detach().cpu().permute(1,2,0).numpy())
        plt.imsave(f'./degrad_ae_128/noisy_{fn}_{global_step}.png', delin_noisy[0].detach().cpu().permute(1,2,0).numpy())


    def log_images(self, train_data, global_step):
        print('Logging a random validation view...')
        for val_idx in range(len(self.val_dataset)):
            if val_idx %  100 != 0:
                continue            
            val_data = self.val_dataset[val_idx]
            val_data = {k : val_data[k][None].to(self.device) if isinstance(val_data[k], torch.Tensor) else val_data[k] for k in val_data.keys()}
            self.log_view_to_tb(global_step, val_data, prefix='val/', fn=f"{global_step}_{val_idx}")
            torch.cuda.empty_cache()

        self.log_view_to_tb(global_step, train_data, prefix=f'train', fn=f'{global_step}')



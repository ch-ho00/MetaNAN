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
from nan.losses import l2_loss
from nan.model import NANScheme
from nan.render_image import render_single_image
from nan.render_ray import RayRender, stack_image, unstack_image
from nan.sample_ray import RaySampler
from nan.utils.eval_utils import mse2psnr, img2psnr
from nan.utils.general_utils import img_HWC2CHW
from nan.utils.io_utils import print_link, colorize
from nan.ssim_l1_loss import MS_SSIM_L1_LOSS
import torch.nn.functional as F
from nan.se3 import SE3_to_se3_N, get_spline_poses
from nan.projection import warp_latent_imgs
from nan.content_loss import reconstruction_loss
from nan.dataloaders.data_utils import get_nearest_pose_ids, get_padded_img_dim, get_depth_warp_img
from nan.restormer_orig import Restormer
import lpips
from skimage.metrics import structural_similarity as ssim
from copy import deepcopy

import random
alpha=0.9999
lpips_fn = lpips.LPIPS(net="vgg")

class Trainer:
    def __init__(self, args):
        if 'objaverse' in args.train_dataset:
            from configs.local_setting_objaverse import OUT_DIR, LOG_DIR
        elif 'deblur' in args.train_dataset:
            from configs.local_setting_deblur import OUT_DIR, LOG_DIR
        elif args.train_dataset == 'llff':
            from configs.local_setting import OUT_DIR, LOG_DIR


        self.args = args
        self.device = torch.device(f"cuda:{args.local_rank}")

        # exp_out_dir will contain .pth weights and .yml config files for the experiment
        self.exp_out_dir = OUT_DIR / args.expname
        self.exp_out_dir.mkdir(exist_ok=True, parents=True)
        print_link(self.exp_out_dir, 'outputs will be saved to')

        # save the args and config files
        self.save_ymls(args, sys.argv[1:], self.exp_out_dir)

        # create training dataset
        if self.args.train_dataset == 'objaverse':
            args.eval_gain = [4,2,1]
        else:
            if self.args.eval_mode:
                args.eval_gain = [1,2,4,8,16,20]
            else:
                args.eval_gain = [20,16,8]
        self.train_dataset, self.train_sampler = create_training_dataset(args)
        # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
        # please use distributed parallel on multiple GPUs to train multiple target views per batch
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1,
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
        self.model = NANScheme.create(args)
        
        if self.args.clean_ckpt_path != None:
            clean_model_args = deepcopy(args)
            clean_model_args.ckpt_path = Path(args.clean_ckpt_path)
            self.clean_model = NANScheme.create(clean_model_args)
            self.clean_model_args = clean_model_args
        else:
            self.clean_model = None

        self.pretrain_restormer = None
        if self.args.restormer_ckpt_path:

            states = [self.args.restormer_ckpt_path]
            device_id = torch.cuda.current_device()
            resume_state = torch.load(
                self.args.restormer_ckpt_path,
                map_location=lambda storage, loc: storage.cuda(device_id))

            # network_g:
            #   type: Restormer
            #   inp_channels: 3
            #   out_channels: 3
            #   dim: 48
            #   num_blocks: [4,4,6,6] #[4,6,6,8]
            #   num_refinement_blocks: 3
            #   heads: [1,2,4,6] #[1,2,4,8]
            #   ffn_expansion_factor: 2.0 #2.66
            #   bias: False
            #   LayerNorm_type: WithBias
            #   dual_pixel_task: False
            self.pretrain_restormer = Restormer(
                                inp_channels=3, 
                                out_channels=3, 
                                dim = 48,
                                num_blocks = [4,4,6,6], 
                                num_refinement_blocks = 3,
                                heads = [1,2,4,6],
                                ffn_expansion_factor = 2.0,
                                bias = False,
                                LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
                                dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
            ).cuda()
            self.pretrain_restormer.load_state_dict(resume_state['params'], strict=True)

        # def resume_training(self, resume_state):
        #     """Reload the optimizers and schedulers for resumed training.

        #     Args:
        #         resume_state (dict): Resume state.
        #     """
        #     resume_optimizers = resume_state['optimizers']
        #     resume_schedulers = resume_state['schedulers']
        #     assert len(resume_optimizers) == len(
        #         self.optimizers), 'Wrong lengths of optimizers'
        #     assert len(resume_schedulers) == len(
        #         self.schedulers), 'Wrong lengths of schedulers'
        #     for i, o in enumerate(resume_optimizers):
        #         self.optimizers[i].load_state_dict(o)
        #     for i, s in enumerate(resume_schedulers):
        #         self.schedulers[i].load_state_dict(s)

        # self.pretrain_restormer = 
        self.last_weights_path = None

        # Create ray render object
        self.ray_render = RayRender(model=self.model, args=args, device=self.device)

        # Create criterion
        self.criterion = NANLoss(args)
        self.ssim_alpha = args.ssim_alpha
        self.ssim_l1_loss = MS_SSIM_L1_LOSS(alpha=args.ssim_alpha)

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
        global_step = self.model.start_step + 1
        epoch = 0  # epoch is not consistent when loading ckpt, it affects train_sampler when distributed and prints

        while global_step < self.args.n_iters + 1:
            np.random.seed()
            for train_data in self.train_loader:
                time0 = time.time()
                if self.args.distributed:
                    self.train_sampler.set_epoch(epoch)

                # core optimization loop
                ray_batch_out, ray_batch_in = self.training_loop(train_data, global_step)
                dt = time.time() - time0

                # Logging and saving
                self.logging(train_data, ray_batch_in, ray_batch_out, dt, global_step, epoch)

                global_step += 1
                if global_step > self.model.start_step + self.args.n_iters + 1:
                    break
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

        w = alpha ** global_step
        # Create object that generate and sample rays
        ray_sampler = RaySampler(train_data, self.device)
        N_rand = int(1.0 * self.args.N_rand * self.args.num_source_views / train_data['src_rgbs'][0].shape[0])

        # Sample subset (batch) of rays for the training loop
        ray_batch = ray_sampler.random_ray_batch(N_rand,
                                                 sample_mode=self.args.sample_mode,
                                                 center_ratio=self.args.center_ratio,
                                                 clean=self.args.sup_clean, 
                                                 alpha_sample='objaverse' in self.args.train_dataset)
        # Calculate the feature maps of all views.
        # This step is seperated because in evaluation time we want to do it once for each image.
        H, W = ray_sampler.src_rgbs.shape[2:4]
        if self.args.clean_src_imgs:
            org_src_rgbs = ray_sampler.src_rgbs_clean.to(self.device)
        else:
            org_src_rgbs = ray_sampler.src_rgbs.to(self.device)

        if self.args.burst_length > 1:
            nearby_idxs = []
            src_poses = ray_batch['src_cameras'][:,:,-16:].reshape(-1, 4, 4).cpu().numpy()
            src_intrinsics = ray_batch['src_cameras'][0, :,2:18].reshape(-1, 4, 4)
            for pose in src_poses:
                ids = get_nearest_pose_ids(pose, src_poses, self.args.burst_length, angular_dist_method='dist', sort_by_dist=True)
                nearby_idxs.append(ids)

            nearby_imgs = torch.stack([org_src_rgbs[0].permute(0,3,1,2)[ids] for ids in nearby_idxs])
            nearby_poses = torch.stack([torch.from_numpy(src_poses[ids]) for ids in nearby_idxs]).to(self.device)
            nearby_intrinsics = torch.stack([src_intrinsics[ids] for ids in nearby_idxs])[..., :3,:3]

            tar_pose = ray_batch['camera'][:,-16:].reshape(4, 4).to(self.device)
            tar_intrinsics = ray_batch['camera'][:,2:18].reshape(4, 4).to(self.device)
            tar_near_ids = get_nearest_pose_ids(tar_pose.cpu().numpy(), src_poses, self.args.burst_length - 1, angular_dist_method='dist', sort_by_dist=True)

            tar_nearby_poses = torch.cat([tar_pose[None], torch.from_numpy(src_poses[tar_near_ids]).to(self.device)], dim=0)[None]
            tar_nearby_intrinsics = torch.cat([tar_intrinsics[None], src_intrinsics[tar_near_ids]], dim=0)[..., :3,:3][None]

            tar_noisy_rgb = ray_sampler.rgb.reshape(1,H,W,3).permute(0,3,1,2).to(self.device)
            tar_nearby_imgs = torch.cat([tar_noisy_rgb, org_src_rgbs[0].permute(0,3,1,2)[tar_near_ids]], dim=0)[None]

            nearby_imgs         = torch.cat([tar_nearby_imgs, nearby_imgs], dim=0)
            nearby_poses        = torch.cat([tar_nearby_poses, nearby_poses], dim=0)
            nearby_intrinsics   = torch.cat([tar_nearby_intrinsics, nearby_intrinsics], dim=0)

            extrinsics = torch.inverse(nearby_poses)
            input_imgs = [img for img in nearby_imgs.permute(1,0,2,3,4)]
            depth, _, stage_depths = self.model.patchmatch(input_imgs, nearby_intrinsics, extrinsics, ray_batch['depth_range'][0,0].repeat(self.args.num_source_views+1), ray_batch['depth_range'][0,1].repeat(self.args.num_source_views+1))            
            tar_depth, src_depths = depth[:1], depth[1:]
            
            warped_rgbds, coords = get_depth_warp_img(nearby_imgs[1:], nearby_poses[1:], src_intrinsics, depth[1:].detach(), nearby_idxs)
            ref_rgbd = torch.cat([nearby_imgs[1:][:,:1], src_depths.unsqueeze(1).detach()], dim=2)
            input_imgs = torch.cat([ref_rgbd, warped_rgbds], dim=1)
            reconst_img, feats = self.model.feature_net(input_imgs.reshape(self.args.num_source_views, -1, H, W))
            featmaps = {}
            featmaps['coarse'] = feats[:, :self.args.coarse_feat_dim]
            featmaps['fine']   = feats[:, self.args.coarse_feat_dim:]
            proc_src_rgbs = ray_sampler.src_rgbs.to(self.device)
            org_src_rgbs_ = reconst_img.permute(0,2,3,1)[None] #.detach()
        else:
            reconst_img = None
            tar_depth = None

            org_src_rgbs =  ray_sampler.src_rgbs.to(self.device)
            org_src_rgbs_ = ray_sampler.src_rgbs.to(self.device)

            sigma_est = ray_sampler.sigma_estimate.to(self.device) if ray_sampler.sigma_estimate != None else None
            proc_src_rgbs, featmaps = self.ray_render.calc_featmaps(src_rgbs=org_src_rgbs)

        sigma_est = ray_sampler.sigma_estimate.to(self.device) if ray_sampler.sigma_estimate != None else None
        # Render the rgb values of the pixels that were sampled
        batch_out = self.ray_render.render_batch(ray_batch=ray_batch, 
                                                proc_src_rgbs=proc_src_rgbs, 
                                                featmaps=featmaps,
                                                org_src_rgbs=org_src_rgbs_,
                                                sigma_estimate=sigma_est)
        # compute loss
        torch.cuda.empty_cache()
        self.model.optimizer.zero_grad()
        loss = 0

        coarse_loss         = F.l1_loss(batch_out['coarse'].rgb, ray_batch['rgb'])
        fine_loss           = F.l1_loss(batch_out['fine'].rgb, ray_batch['rgb'])
        loss += coarse_loss + fine_loss
        self.scalars_to_log['train/coarse_loss'] = coarse_loss
        self.scalars_to_log['train/fine_loss'] = fine_loss
        self.scalars_to_log['weight'] = w

        if reconst_img != None:
            clean_src_imgs      = ray_sampler.src_rgbs_clean.to(self.device)[0].permute(0,3,1,2)
            reconst_loss        = F.l1_loss(reconst_img, clean_src_imgs) * 0.01
            loss += reconst_loss 
            self.scalars_to_log['train/reconst_loss'] = reconst_loss
        
        if tar_depth != None:
            xy = ray_batch['xyz'][:,:2]
            pseudo_depth = (batch_out['fine'].depth + batch_out['coarse'].depth) / 2
            depth_loss = 0

            coords = xy.clone().float()
            coords[:, 0] = (coords[:, 0] / W ) * 2 - 1  
            coords[:, 1] = (coords[:, 1] / H ) * 2 - 1  
            for k in stage_depths.keys():
                for depth in stage_depths[k]:
                    sel_ref_depth = F.grid_sample(depth[:1], coords[None, None].cuda()).squeeze()
                    depth_loss += F.smooth_l1_loss(sel_ref_depth, pseudo_depth.detach()) * 0.01
            loss += depth_loss
            self.scalars_to_log['train/depth_loss'] = depth_loss

        loss.backward()
        self.scalars_to_log['loss'] = loss.item()

        if self.args.burst_length > 1:
            torch.nn.utils.clip_grad_norm_(self.model.patchmatch.parameters(), 0.01)
            torch.nn.utils.clip_grad_norm_(self.model.feature_net.parameters(), 0.01)

        self.model.optimizer.step()
        self.model.scheduler.step()
        self.model.optimizer.zero_grad()

        self.scalars_to_log['lr_features'] = self.model.scheduler.get_last_lr()[0]
        self.scalars_to_log['lr_mlp'] = self.model.scheduler.get_last_lr()[1]
        del proc_src_rgbs, featmaps, ray_sampler
        return batch_out, ray_batch

    def logging(self, train_data, ray_batch_in, ray_batch_out, dt, global_step, epoch, max_keep=10):
        if self.args.local_rank == 0:
            # log iteration values
            if global_step % self.args.i_tb == 0 or global_step < 10:
                self.log_iteration(ray_batch_out, ray_batch_in, dt, global_step, epoch)

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
            if global_step % self.args.i_img == 0 or global_step == self.model.start_step + 1:
                self.log_images(train_data, global_step)

    def log_iteration(self, ray_batch_out, ray_batch_in, dt, global_step, epoch):
        # write mse and psnr stats
        if ray_batch_in['white_level'] != None:
            mse_error = l2_loss(de_linearize(ray_batch_out['coarse'].rgb, ray_batch_in['white_level']).clamp(0,1),
                                de_linearize(ray_batch_in['rgb'], ray_batch_in['white_level']).clamp(0,1)).item()
        else:
            mse_error = l2_loss(ray_batch_out['coarse'].rgb.clamp(0,1),
                                ray_batch_in['rgb'].clamp(0,1)).item()

        self.scalars_to_log['train/coarse-loss'] = mse_error
        self.scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
        if ray_batch_out['fine'] is not None:
            if ray_batch_in['white_level'] != None:
                mse_error = l2_loss(de_linearize(ray_batch_out['fine'].rgb, ray_batch_in['white_level']).clamp(0,1),
                                    de_linearize(ray_batch_in['rgb'], ray_batch_in['white_level']).clamp(0,1)).item()
            else:
                mse_error = l2_loss(ray_batch_out['fine'].rgb.clamp(0,1),
                                    ray_batch_in['rgb'].clamp(0,1)).item()

            self.scalars_to_log['train/fine-loss'] = mse_error
            self.scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)

        logstr = f"{self.args.expname} Epoch: {epoch}  step: {global_step} "
        for k in self.scalars_to_log.keys():
            logstr += f" {k}: {self.scalars_to_log[k]:.6f}"
            self.writer.add_scalar(k, self.scalars_to_log[k], global_step)
        if global_step % self.args.i_print == 0:
            print(logstr)
            print(f"each iter time {dt:.05f} seconds")

    def log_view_to_tb(self, global_step, ray_sampler, gt_img, render_stride=1, prefix='', postfix='', visualize=False):
        self.model.switch_to_eval()
        with torch.no_grad():

            if self.pretrain_restormer != None:
                ray_sampler.denoised_src_rgbs = self.pretrain_restormer(ray_sampler.src_rgbs[0].permute(0,3,1,2).cuda()).permute(0,2,3,1)[None]
            else:
                ray_sampler.denoised_src_rgbs = None

            start = time.time()
            ret = render_single_image(ray_sampler=ray_sampler, model=self.model, args=self.args, global_step=global_step, denoised_input= self.pretrain_restormer != None and self.clean_model == None) # if both none, analyze the depth and reconst psnr
            print("One image rendering time =", round(time.time() - start, 3))
            if self.clean_model != None:
                clean_ret = render_single_image(ray_sampler=ray_sampler, model=self.clean_model, args=self.clean_model_args, global_step=global_step, clean_src_imgs=True)
        if self.args.clean_src_imgs:
            average_im = ray_sampler.src_rgbs_clean.cpu()[0,0]        
        else:
            average_im = ray_sampler.src_rgbs.cpu()[0,0]

        if self.args.render_stride != 1:
            gt_img = gt_img[::render_stride, ::render_stride]
            average_im = average_im[::render_stride, ::render_stride]
            reconst_signal = None
        
        rgb_gt = img_HWC2CHW(gt_img)
        average_im = img_HWC2CHW(average_im)

        rgb_pred = img_HWC2CHW(ret['coarse'].rgb.detach().cpu())
        if visualize:
            h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
            w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
            rgb_im = torch.zeros(3, h_max, 3 * w_max)
            rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
            rgb_im[:, :rgb_gt.shape[-2], w_max:w_max + rgb_gt.shape[-1]] = rgb_gt
            rgb_im[:, :rgb_pred.shape[-2], 2 * w_max:2 * w_max + rgb_pred.shape[-1]] = rgb_pred

            depth_im = ret['coarse'].depth.detach().cpu()
            # import pdb; pdb.set_trace()
            acc_map = torch.sum(ret['coarse'].weights, dim=-1).detach().cpu()

            if ret['fine'] is None:
                depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
                acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
            else:
                rgb_fine = img_HWC2CHW(ret['fine'].rgb.detach().cpu())
                rgb_fine_ = torch.zeros(3, h_max, w_max)
                rgb_fine_[:, :rgb_fine.shape[-2], :rgb_fine.shape[-1]] = rgb_fine
                rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
                # rgb_im = rgb_im
                if ray_sampler.white_level != None:
                    rgb_im = de_linearize(rgb_im, ray_sampler.white_level).clamp(min=0., max=1.)
                else:
                    rgb_im = rgb_im.clamp(min=0., max=1.)
                depth_im = torch.cat((depth_im, ret['fine'].depth.detach().cpu()), dim=-1)
                depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
                acc_map = torch.cat((acc_map, torch.sum(ret['fine'].weights, dim=-1).detach().cpu()), dim=-1)
                acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

            # write the pred/gt rgb images and depths
            self.writer.add_image(prefix + 'rgb_gt-coarse-fine' + postfix, rgb_im, global_step)
            self.writer.add_image(prefix + 'depth_gt-coarse-fine'+ postfix, depth_im, global_step)
            self.writer.add_image(prefix + 'acc-coarse-fine'+ postfix, acc_map, global_step)
            h,w = list(ray_sampler.src_rgbs.shape[-3:-1])


            if 'kernel_reconst' in ret.keys():
                vis_img = torch.cat([ray_sampler.src_rgbs, ret['kernel_reconst'].permute(0,2,3,1)[None].cpu(), ray_sampler.src_rgbs_clean], dim=0)
                vis_img = vis_img[:,:2]
                vis_img = vis_img.permute(4,1,2,0,3).reshape(3, h * 2, -1)[:, ::render_stride, ::render_stride]
                vis_img = vis_img.cpu().clamp(0,1)

                if ray_sampler.white_level != None:
                    vis_imgs = de_linearize(vis_img.cpu(), ray_sampler.white_level).clamp(0,1)

                self.writer.add_image(prefix + 'kernel_reconst'+ postfix, vis_img, global_step)

            if 'patchmatch_depth' in ret.keys():
                pred_depth = ret['patchmatch_depth'].squeeze().permute(1,0,2).reshape(h,-1)[::render_stride, ::render_stride]
                pred_depth = img_HWC2CHW(colorize(pred_depth, cmap_name='jet', append_cbar=True))
                self.writer.add_image(prefix + 'patchmatch_depth'+ postfix, pred_depth, global_step)

            if 'depth_warped_imgs' in ret.keys():
                vis_imgs = ret['depth_warped_imgs'].permute(2,0,3,1,4).reshape(3, h * 2, -1)[:, ::render_stride, ::render_stride]
                if ray_sampler.white_level != None:
                    vis_imgs = de_linearize(vis_img.cpu(), ray_sampler.white_level).clamp(0,1)

                self.writer.add_image(prefix + 'depth_warped_imgs'+ postfix, vis_imgs, global_step)

            del depth_im, rgb_im
        # write scalar
        pred_rgb = ret['fine'].rgb if ret['fine'] is not None else ret['coarse'].rgb
        if ray_sampler.white_level != None:
            pred_rgb = de_linearize(pred_rgb.detach().cpu(), ray_sampler.white_level).clamp(0,1)
            gt_img   = de_linearize(gt_img, ray_sampler.white_level).clamp(0,1)

        pred_img = pred_rgb.detach().cpu().clamp(0,1)
        gt_img = gt_img.clamp(0,1)
        psnr_curr_img = img2psnr(pred_img, gt_img)
        ssim_curr_img = ssim( pred_img.numpy(), gt_img.cpu().numpy(), data_range=1, multichannel=True, channel_axis=-1)
        lpips_curr_img = lpips_fn(pred_img.permute(2,0,1)[None] * 2 - 1, gt_img.permute(2,0,1)[None] * 2 - 1).item()  # Normalize to [-1,1]
        print(psnr_curr_img, ssim_curr_img, lpips_curr_img, "##")
        out = [psnr_curr_img, ssim_curr_img, lpips_curr_img]
        if self.clean_model != None:
            clean_depth = clean_ret['fine'].depth
            pred_depth = ret['fine'].depth
            near, far = ray_sampler.depth_range.squeeze().numpy()
            out += [[(clean_depth - pred_depth).abs().mean().item(), (clean_depth - pred_depth).abs().mean().item() / (far - near)]]
            print("Depth offset = ", out[-1])
        self.model.switch_to_train()
        del pred_rgb, ret

        return out

    def log_images(self, train_data, global_step):
        print('Logging a random validation view...')
        cnt = 0
        torch.cuda.empty_cache()
        psnr_results = {}
        psnr_scene_results = {}
        depth_err_results = {}
        vis_interval = 4 if self.args.ckpt_path == None else 1
        print("Size of Val dataset = ", len(self.val_dataset))
        for val_idx in range(len(self.val_dataset)):

            if global_step == 1 and val_idx > 0 and not self.args.eval_mode:
                break

            visualize = False #True if val_idx % vis_interval == 0 else False
            cnt += 1 
            val_data = self.val_dataset[val_idx]
            val_data = {k : val_data[k][None] if isinstance(val_data[k], torch.Tensor) else val_data[k] for k in val_data.keys()}
            tmp_ray_sampler = RaySampler(val_data, self.device, render_stride=self.args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            gt_img = tmp_ray_sampler.rgb_clean.reshape(H, W, 3)
            eval_gain = val_data['eval_gain']
            blur_level = val_data['rgb_path'].split('/')[-2]
            blur_level = blur_level if 'mix' not in blur_level else 'blur_mix'
            if self.args.add_burst_noise:
                blur_level = blur_level + f"_gain{val_data['eval_gain']}"
            out = self.log_view_to_tb(global_step, tmp_ray_sampler, gt_img, render_stride=self.args.render_stride, prefix='val/', postfix=f"_gain{eval_gain}_iter{cnt}", visualize=visualize)
            psnr, ssim, lpips = out[:3]
            if self.clean_model != None:
                abs_depth_err, rel_depth_err = out[-1]
                
            print("@@@@@@@@", psnr, ssim, lpips)
            if eval_gain in psnr_results.keys():
                psnr_results[eval_gain].append(psnr)
            else:
                psnr_results[eval_gain] = [psnr]

            if self.args.train_dataset == 'objaverse': #and self.args.eval_dataset == 'deblur_test':
                if blur_level + "_psnr" in psnr_scene_results.keys():
                    psnr_scene_results[blur_level + "_psnr"] += [psnr]
                    psnr_scene_results[blur_level + "_ssim"] += [ssim]
                    psnr_scene_results[blur_level + "_lpips"] += [lpips]
                    if self.clean_model != None:
                        psnr_scene_results[blur_level + "_clean_abs_depth_err"] += [abs_depth_err]
                        psnr_scene_results[blur_level + "_clean_rel_depth_err"] += [rel_depth_err]

                else:
                    psnr_scene_results[blur_level + "_psnr"] = [psnr]
                    psnr_scene_results[blur_level + "_ssim"] = [ssim]
                    psnr_scene_results[blur_level + "_lpips"] = [lpips]
                    if self.clean_model != None:
                        psnr_scene_results[blur_level + "_clean_abs_depth_err"] = [abs_depth_err]
                        psnr_scene_results[blur_level + "_clean_rel_depth_err"] = [rel_depth_err]
            # print(psnr_scene_results)
            del tmp_ray_sampler, val_data, gt_img 
            torch.cuda.empty_cache()
            print("val image #",cnt)

        for k in psnr_results.keys():
            self.writer.add_scalar('val/' + f'psnr_gain{k}', np.mean(psnr_results[k]), global_step)
            for idx, psnr in enumerate(psnr_results[k]):
                self.writer.add_scalar('val/' + f'psnr_gain{k}/img{idx}', psnr, global_step)

        if len(psnr_scene_results.keys()) > 0:
            for k in psnr_scene_results.keys():
                self.writer.add_scalar('val/' + f'{k}/', np.mean(psnr_scene_results[k]), global_step)

        print('Logging current training view...')
        tmp_ray_train_sampler = RaySampler(train_data, self.device,
                                           render_stride=self.args.render_stride)
        H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
        gt_img = tmp_ray_train_sampler.rgb_clean.reshape(H, W, 3)
        self.log_view_to_tb(global_step, tmp_ray_train_sampler, gt_img, render_stride=self.args.render_stride, prefix='train/')
        del tmp_ray_train_sampler 
        torch.cuda.empty_cache()
        import pdb; pdb.set_trace()
        if self.args.eval_mode:
            exit()




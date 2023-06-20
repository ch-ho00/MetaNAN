# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# #### Modified version of LLFF dataset code
# #### see https://github.com/googleinterns/IBRNet for original
import sys
from abc import ABC
from pathlib import Path

import imageio
import numpy as np
import torch
import random, math

from configs.local_setting import LOG_DIR
from nan.dataloaders.basic_dataset import NoiseDataset, re_linearize
from nan.dataloaders.data_utils import random_crop, get_nearest_pose_ids, random_flip, to_uint
from nan.dataloaders.llff_data_utils import load_llff_data, batch_parse_llff_poses
from nan.dataloaders.basic_dataset import Mode


from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
import itertools


class SeaNeRFDataset(NoiseDataset, ABC):
    name = 'seanerf'

    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_ids = []
        self.render_depth_range = []

        self.src_intrinsics = []
        self.src_poses = []
        self.src_rgb_files = []
        super().__init__(args, mode, scenes=scenes, random_crop=random_crop, **kwargs)
        self.depth_range = self.render_depth_range[0]

        # blur settings for the first degradation
        self.blur_kernel_size = args.blur_kernel_size
        self.kernel_list = args.kernel_list
        self.kernel_prob = args.kernel_prob  # a list for each kernel probability
        self.blur_sigma = args.blur_sigma
        self.betag_range = args.betag_range  # betag used in generalized Gaussian blur kernels
        self.betap_range = args.betap_range  # betap used in plateau blur kernels
        self.sinc_prob = args.sinc_prob  # the probability for sinc filters
        self.jpeg_range = args.jpeg_range
        self.blur_degrade = args.blur_degrade
        
        # a final sinc filter
        self.final_sinc_prob = args.final_sinc_prob
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts        

    def get_i_test(self, N):
        return np.arange(N)[::self.args.llffhold] 

    def get_i_train(self, N, i_test, mode):
        return np.array([j for j in np.arange(int(N)) if j not in i_test]) 

    @staticmethod
    def load_scene(scene_path, factor):
        return load_llff_data(scene_path, load_imgs=False, factor=factor, seanerf=True)

    def __len__(self):
        return len(self.render_rgb_files) * 100000 if self.mode is Mode.train else len(self.render_rgb_files) 

    def set_noise_param_list(self, N_sample=4):

        eps = 0.001
        self.omega_c_list = [ [tick - eps, tick] for tick in np.linspace(np.pi / 3,  np.pi, N_sample, endpoint=True)]
        self.kernel_list = ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"]
        self.blur_sigma_list  = [ [tick, tick + eps] for tick in np.linspace(0.2, 3, N_sample, endpoint=True)] 
        self.betag_range_list = [ [tick, tick + eps] for tick in np.linspace(0.5, 4, N_sample, endpoint=True)  ]
        self.betap_range_list = [ [tick, tick + eps] for tick in np.linspace(1, 2, N_sample, endpoint=True)  ]
        self.all_combination = list(itertools.product(self.omega_c_list, self.kernel_list, self.blur_sigma_list, self.betag_range_list, self.betap_range_list))
          

    def get_singleview_param(self, idx=None, hparams=None, eval_gain=1):

        if idx != None:
            hparams = self.all_combination[idx]
        else:
            assert hparams != None    
        
        omega_c, kernel, blur_sigma, betag_range, bet_ap_range = hparams      


        # idx = idx // len(self.all_combination)
        idx = 0
        rgb_file: Path = self.render_rgb_files[idx]
        # image (H, W, 3)
        rgb = self.read_image(rgb_file)

        side = self.args.img_size
        crop_h = 768 // 2
        crop_w = 1024 // 2
        rgb = rgb[crop_h:crop_h+side, crop_w:crop_w+side].transpose((2,0,1))[None]
        white_level = torch.Tensor([1])

        # d1
        rgb_d1 = self.apply_blur_kernel(torch.from_numpy(rgb), params=hparams).clamp(0,1)
        rgb_d1 = re_linearize(rgb_d1, white_level)
        rgb_d1, _ = self.add_noise_level(rgb_d1, eval_gain)                        


        norm_omega_c = (np.mean(hparams[0]) - np.pi / 3) / (np.pi - np.pi / 3)
        kernel = hparams[1]
        norm_sigma = (np.mean(hparams[2]) - 0.2) / (3 - 0.2)
        norm_betag = (np.mean(hparams[3]) - 0.5) / (4 - 0.5)
        norm_betap = (np.mean(hparams[4]) - 1) / (2 - 1)
        # self.omega_c_list = [ [tick - eps, tick] for tick in np.linspace(np.pi / 3,  np.pi, N_sample, endpoint=True)]
        # self.kernel_list = ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"]
        # self.blur_sigma_list  = [ [tick, tick + eps] for tick in np.linspace(0.2, 3, N_sample, endpoint=True)] 
        # self.betag_range_list = [ [tick, tick + eps] for tick in np.linspace(0.5, 4, N_sample, endpoint=True)  ]
        # self.betap_range_list = [ [tick, tick + eps] for tick in np.linspace(1, 2, N_sample, endpoint=True)  ]

        parsed_hparams = {
            'omega_c'      : round(norm_omega_c,3), 
            'kernel'       : kernel, 
            'blur_sigma'   : round(norm_sigma,3), 
            'betag_range'  : round(norm_betag,3), 
            'betap_range' : round(norm_betap,3),
            'eval_gain'    : eval_gain,
            'white_level'  : white_level
        }

        return rgb_d1, parsed_hparams    

    def __getitem__(self, idx):
        if self.args.degae_training:
            return self.get_singleview_item(idx)            
        else:
            return self.get_multiview_item(idx)

    def apply_blur_kernel(self, rgb, final_sinc=False, params=None):
        kernel_size = random.choice(self.kernel_range)
        rand_params = True
        if params != None:
            omega_c, kernel, blur_sigma, betag_range, betap_range = params
            rand_params = False
        if not final_sinc:
            if np.random.uniform() < self.args.sinc_prob:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list if rand_params else [kernel],
                    self.kernel_prob if rand_params else [1],
                    kernel_size,
                    self.blur_sigma if rand_params else blur_sigma,
                    self.blur_sigma if rand_params else blur_sigma, [-math.pi, math.pi],
                    self.betag_range if rand_params else betag_range,
                    self.betap_range if rand_params else betap_range,
                    noise_range=None)

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
            kernel = kernel.astype(np.float32)
            out = filter2D(rgb, torch.from_numpy(kernel[None].repeat(rgb.shape[0], 0)))

        else:
            # ------------------------------------- the final sinc kernel ------------------------------------- #
            if np.random.uniform() < self.args.final_sinc_prob:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = sinc_kernel.astype(np.float32)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor
            out = filter2D(rgb, sinc_kernel[None].repeat(rgb.shape[0], 1, 1))

        return out
    
    def apply_jpeg_compression(self, rgb):
        # JPEG compression
        jpeg_p = rgb.new_zeros(rgb.size(0)).uniform_(*self.args.jpeg_range)
        rgb = torch.clamp(rgb, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(rgb, quality=jpeg_p)
        
        return out 
    
    def get_singleview_item(self, idx):
        # Read target data:
        eval_gain = self.args.eval_gain[idx // len(self.render_rgb_files)]
        idx = idx % len(self.render_rgb_files)
        rgb_file: Path = self.render_rgb_files[idx]
        # image (H, W, 3)
        rgb = self.read_image(rgb_file)

        side = self.args.img_size
        if self.mode in [Mode.train]:
            crop_h = np.random.randint(low=0, high=768 - side)
            crop_w =  np.random.randint(low=0, high=1024 - side)
        else:
            crop_h = 768 // 2
            crop_w = 1024 // 2
        rgb = rgb[crop_h:crop_h+side, crop_w:crop_w+side].transpose((2,0,1))[None]

        idx_ref = idx
        while idx == idx_ref:
            idx_ref = random.choice(list(range(len(self.render_rgb_files))))        
        rgb_file_ref: Path = self.render_rgb_files[idx_ref]
        rgb_ref = self.read_image(rgb_file_ref)

        crop_h = np.random.randint(low=0, high=768 - side)
        crop_w =  np.random.randint(low=0, high=1024 -side)
        rgb_ref = rgb_ref[crop_h:crop_h+side, crop_w:crop_w+side].transpose((2,0,1))[None]

        if self.mode in [Mode.train]:
            if random.random() < 0.5:
                rgb = np.flip(rgb, axis=-1).copy()
            if random.random() < 0.5:
                rgb = np.flip(rgb, axis=-2).copy()

            if random.random() < 0.5:
                rgb_ref = np.flip(rgb_ref, axis=-1).copy()
            if random.random() < 0.5:
                rgb_ref = np.flip(rgb_ref, axis=-2).copy()

            white_level = torch.clamp(10 ** -torch.rand(1), 0.6, 1)
        else:
            white_level = torch.Tensor([1])

        # d1
        if self.mode is Mode.train:
            if self.blur_degrade and random.random() > 0.25:
                rgb_d1 = self.apply_blur_kernel(torch.from_numpy(rgb), final_sinc=False).clamp(0,1)
            else:
                rgb_d1 = rgb 
            rgb_d1 = re_linearize(rgb_d1, white_level)
            clean_d1 = False

            if random.random() > 0.25:
                rgb_d1 , _ = self.add_noise(rgb_d1)
            else:
                clean_d1 = True        
            
            # if random.random() < self.final_sinc_prob and self.blur_degrade:
            #     rgb_d1 = self.apply_blur_kernel(rgb_d1, final_sinc=True)
                
        else:
            rgb_d1 = re_linearize(rgb, white_level)
            rgb_d1, _ = self.add_noise_level(rgb_d1, eval_gain)                        

        # d2
        d2_rgbs = np.concatenate([rgb, rgb_ref], axis=0)
        d2_rgbs = torch.from_numpy(d2_rgbs)
        if self.mode is Mode.train:
            if self.blur_degrade and random.random() > 0.25:
                d2_rgbs = self.apply_blur_kernel(d2_rgbs, final_sinc=False).clamp(0,1)
            d2_rgbs = re_linearize(d2_rgbs, white_level)
            if random.random() > 0.25 or clean_d1:
                d2_rgbs, _ = self.add_noise(d2_rgbs)
                clean_d2 = False        

            # if random.random() < self.final_sinc_prob and self.blur_degrade:
            #     d2_rgbs = self.apply_blur_kernel(d2_rgbs, final_sinc=True)
        else:
            d2_rgbs = re_linearize(d2_rgbs[:, :3], white_level)

        rgb_d2, rgb_ref_d2 = d2_rgbs[0], d2_rgbs[1]
        batch_dict = {
                      'noisy_rgb'       : rgb_d1.squeeze(),
                      'clean_rgb'       : torch.from_numpy(rgb).squeeze(),
                      'target_rgb'      : rgb_d2.squeeze(),
                      'ref_rgb'         : rgb_ref_d2.squeeze(),
                      'white_level'     : white_level
        }


        if self.mode is not Mode.train:
            batch_dict['eval_gain'] = eval_gain

        return batch_dict        

    def get_multiview_item(self, idx):
        # Read target data:
        eval_gain = -1 #self.args.eval_gain[idx // len(self.render_rgb_files)]
        idx = idx % len(self.render_rgb_files)
        rgb_file: Path = self.render_rgb_files[idx]
        

        # image (H, W, 3)
        rgb = self.read_image(rgb_file)

        # Rotation | translation (4x4)
        # 0  0  0  | 1
        render_pose = self.render_poses[idx]

        # K       (4x4)
        # 0 0 0 1
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]
        camera = self.create_camera_vector(rgb, intrinsics, render_pose) # shape: 34 (H, W, K, R|t)

        # Read src data:
        train_set_id = self.render_train_set_ids[idx] # scene number
        train_rgb_files = self.src_rgb_files[train_set_id] # N optional src files in the scene
        train_poses = self.src_poses[train_set_id]  # (N, 4, 4)
        train_intrinsics = self.src_intrinsics[train_set_id]  # (N, 4, 4)

        if self.mode is Mode.train:
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = None
            subsample_factor = 1 #np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views # + np.random.randint(low=-2, high=self.num_select_high)
            id_render = id_render
        else:
            id_render = None
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = self.get_nearest_pose_ids(render_pose, depth_range, train_poses, subsample_factor, id_render)
        nearest_pose_ids = self.choose_views(nearest_pose_ids, num_select, id_render)

        src_rgbs = []
        src_cameras = []
        for src_id in nearest_pose_ids:
            if src_id is None:
                # print(self.render_rgb_files[idx])
                src_rgb = self.read_image(self.render_rgb_files[idx])
                train_pose = self.render_poses[idx]
                train_intrinsics_ = self.render_intrinsics[idx]
            else:
                # print(train_rgb_files[src_id])
                src_rgb = self.read_image(train_rgb_files[src_id])
                train_pose = train_poses[src_id]
                train_intrinsics_ = train_intrinsics[src_id]

            src_rgbs.append(src_rgb)
            src_camera = self.create_camera_vector(src_rgb, train_intrinsics_, train_pose)

            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0) # (num_select, H, W, 3)
        src_cameras = np.stack(src_cameras, axis=0) # (num_select, 34)

        rgb, camera, src_rgbs, src_cameras = self.apply_transform(rgb, camera, src_rgbs, src_cameras)
        gt_depth = 0

        depth_range = self.final_depth_range(depth_range)
        return self.create_seanerf_batch_from_numpy(rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range, gt_depth=gt_depth, eval_gain=eval_gain)

    def get_nearest_pose_ids(self, render_pose, depth_range, train_poses, subsample_factor, id_render):
        return get_nearest_pose_ids(render_pose,
                                    train_poses,
                                    min(self.num_source_views * subsample_factor, self.min_nearest_pose),
                                    tar_id=id_render,
                                    angular_dist_method='dist')

    def add_single_scene(self, i, scene_path):
        _, poses, bds, _, i_test, rgb_files = self.load_scene(scene_path, None)
        near_depth = bds.min()
        far_depth = bds.max()


        sh = imageio.imread(rgb_files[0]).shape
        sh = [int(side * 0.6) for side in sh[:2]]
        poses[:, :2, 4] = np.array(sh[:2]).reshape([2])
        poses[:,  2, 4] = poses[:, 2, 4] * 0.6    

        intrinsics, c2w_mats = batch_parse_llff_poses(poses, hw=sh)

        i_test = self.get_i_test(poses.shape[0])
        i_train = self.get_i_train(poses.shape[0], i_test, self.mode)

        if self.mode is Mode.train:
            i_render = i_train
        else:
            i_render = i_test

        # Source images
        self.src_intrinsics.append(intrinsics[i_train])
        self.src_poses.append(c2w_mats[i_train])
        self.src_rgb_files.append([rgb_files[i] for i in i_train])

        # Target images
        num_render = len(i_render)
        self.render_rgb_files.extend([rgb_files[i] for i in i_render])
        self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
        self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
        self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
        self.render_train_set_ids.extend([i] * num_render)
        self.render_ids.extend(i_render)



class SeaNeRFTestDataset(SeaNeRFDataset):
    name = 'seanerf_test'
    dir_name = 'SeathruNeRF_dataset'
    num_select_high = 2
    min_nearest_pose = 28

    def apply_transform(self, rgb, camera, src_rgbs, src_cameras):
        if self.mode is Mode.train and self.random_crop:
            crop_h = np.random.randint(low=250, high=750) // 128 * 128
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(350 * 550 / crop_h // 128 * 128)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                             (crop_h, crop_w))

        if self.mode is Mode.train and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        return rgb, camera, src_rgbs, src_cameras

    def final_depth_range(self, depth_range):
        return torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])


class SeaNeRFTrainDataset(SeaNeRFTestDataset):
    name = 'seanerf'
    dir_name = 'SeathruNeRF_dataset'
    num_select_high = 3
    min_nearest_pose = 20

    def __len__(self):
        return len(self.render_rgb_files)


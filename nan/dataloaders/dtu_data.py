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
import os
import re
import cv2

from configs.local_setting import LOG_DIR
from nan.dataloaders.basic_dataset import NoiseDataset, re_linearize
from nan.dataloaders.data_utils import random_crop, get_nearest_pose_ids, random_flip, to_uint
from nan.dataloaders.llff_data_utils import load_llff_data, batch_parse_llff_poses
from nan.dataloaders.basic_dataset import Mode
from nan.dataloaders.llff_data_utils import recenter_poses

from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


class DTUDataset(NoiseDataset, ABC):
    name = 'colmap'

    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.render_rgb_files = []
        self.render_depth_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_ids = []
        self.render_depth_range = []

        self.src_intrinsics = []
        self.src_poses = []
        self.src_rgb_files = []
        self.src_depth_files = []
        self.scale_factor = 1 / 200.0 
        self.img_wh = [640, 512]
        self.opencv2blender = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
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
        return np.arange(N)[::self.args.llffhold] if not self.args.degae_training else  np.array([j for j in np.arange(int(N))]) 

    def get_i_train(self, N, i_test, mode):
        return np.array([j for j in np.arange(int(N)) if j not in i_test]) if not self.args.degae_training else  np.array([j for j in np.arange(int(N))]) 

    @staticmethod
    def load_scene(scene_path, factor):
        return load_llff_data(scene_path, load_imgs=False, factor=factor)

    def __len__(self):
        if self.args.degae_training:
            return 1000 if self.mode is Mode.train else len(self.render_rgb_files) * len(self.args.eval_gain)
        else:
            return len(self.render_rgb_files) * 100000 if self.mode is Mode.train else len(self.render_rgb_files) * len(self.args.eval_gain)

    def __getitem__(self, idx):
        if self.args.degae_training:
            return self.get_singleview_item(idx)            
        else:
            return self.get_multiview_item(idx)

    def apply_blur_kernel(self, rgb, final_sinc=False):

        kernel_size = random.choice(self.kernel_range)
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
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
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
        rgb = self.read_image(rgb_file, img_wh=self.img_wh)

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
        rgb_ref = self.read_image(rgb_file_ref, img_wh=self.img_wh)

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
            if self.blur_degrade:
                rgb_d1 = self.apply_blur_kernel(torch.from_numpy(rgb), final_sinc=False).clamp(0,1)
            else:
                rgb_d1 = rgb 
            rgb_d1 = re_linearize(rgb_d1, white_level)
            clean_d1 = False

            if random.random() > 0.25:
                rgb_d1 , _ = self.add_noise(rgb_d1)
            else:
                clean_d1 = True        
            
            # if random.random() < self.final_sinc_prob:
            #     rgb_d1 = self.apply_blur_kernel(rgb_d1, final_sinc=True)
                
        else:
            rgb_d1 = re_linearize(rgb, white_level)
            rgb_d1, _ = self.add_noise_level(rgb_d1, eval_gain)                        

        # d2
        d2_rgbs = np.concatenate([rgb, rgb_ref], axis=0)
        d2_rgbs = torch.from_numpy(d2_rgbs)
        if self.mode is Mode.train:
            if self.blur_degrade:
                d2_rgbs = self.apply_blur_kernel(d2_rgbs, final_sinc=False).clamp(0,1)
            d2_rgbs = re_linearize(d2_rgbs, white_level)
            if random.random() > 0.25 or clean_d1:
                d2_rgbs, _ = self.add_noise(d2_rgbs)        

            # if random.random() < self.final_sinc_prob:
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

        if rgb_d1.isnan().sum() + torch.from_numpy(rgb).isnan().sum() + rgb_d2.isnan().sum() + rgb_ref_d2.isnan().sum() > 0:
            import pdb; pdb.set_trace()

        if self.mode is not Mode.train:
            batch_dict['eval_gain'] = eval_gain

        return batch_dict        

    def get_multiview_item(self, idx):
        # Read target data:
        eval_gain = self.args.eval_gain[idx // len(self.render_rgb_files)]
        idx = idx % len(self.render_rgb_files)
        rgb_file: Path = self.render_rgb_files[idx]
        depth_file: Path = self.render_depth_files[idx]

        # image (H, W, 3)
        rgb = self.read_image(rgb_file, img_wh=self.img_wh)
        _, _, depth_h = self.read_depth(depth_file)
        depth_h *= self.scale_factor
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
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
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
                src_rgb = self.read_image(self.render_rgb_files[idx], img_wh=self.img_wh)
                train_pose = self.render_poses[idx]
                train_intrinsics_ = self.render_intrinsics[idx]
            else:
                # print(train_rgb_files[src_id])
                src_rgb = self.read_image(train_rgb_files[src_id], img_wh=self.img_wh)
                train_pose = train_poses[src_id]
                train_intrinsics_ = train_intrinsics[src_id]

            src_rgbs.append(src_rgb)
            src_camera = self.create_camera_vector(src_rgb, train_intrinsics_, train_pose)

            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0) # (num_select, H, W, 3)
        src_cameras = np.stack(src_cameras, axis=0) # (num_select, 34)

        rgb, camera, src_rgbs, src_cameras = self.apply_transform(rgb, camera, src_rgbs, src_cameras)

        # Load reference Image
        ref_rgb = None
        if self.args.degae_feat and self.args.meta_module:
            ref_idx = idx
            ref_set_id = self.render_train_set_ids[ref_idx] # scene number
            # choose from another scene
            while ref_set_id == train_set_id:
                ref_idx = random.choice(list(range(len(self.render_rgb_files))))
                ref_set_id = self.render_train_set_ids[ref_idx] # scene number
            
            ref_rgb_files = self.src_rgb_files[ref_set_id] # N optional src files in the scene
            ref_img_file = random.choice(ref_rgb_files)
            ref_rgb = self.read_image(ref_img_file, img_wh=self.img_wh)
            side = self.args.img_size
            crop_h = np.random.randint(low=0, high=768 - side)
            crop_w =  np.random.randint(low=0, high=1024 - side)
            ref_rgb = ref_rgb[crop_h:crop_h+side, crop_w:crop_w+side]

        depth_range = torch.tensor(depth_range) #torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])
        gt_depth = None
        return self.create_batch_from_numpy(rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range, gt_depth=gt_depth, eval_gain=eval_gain, ref_rgb=ref_rgb)

    def get_nearest_pose_ids(self, render_pose, depth_range, train_poses, subsample_factor, id_render):
        return get_nearest_pose_ids(render_pose,
                                    train_poses,
                                    min(self.num_source_views * subsample_factor, self.min_nearest_pose),
                                    tar_id=id_render,
                                    angular_dist_method='dist')


    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor
        self.depth_interval = float(lines[11].split()[1])
        return intrinsics, extrinsics, [depth_min, depth_max]

    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=1.0, fy=1.0,
                             interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        depth = cv2.resize(depth_h, None, fx=1.0 / 4, fy=1.0 / 4,
                           interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        mask = depth > 0

        return depth, mask, depth_h


    def load_dtu_scene(self, scene_path):

        rgb_files, depth_files = [], []
        intrinsics, c2ws, near_fars = [], [], [] 
        light_idx = 3
        view_ids = range(49)
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            scene_folder_name = scene_path.split("/")[-1]
            img_filename = os.path.join(scene_path,
                                        f'rect_{vid + 1:03d}_{light_idx}_r5000.png')
            depth_filename = os.path.join("/".join(scene_path.split("/")[:-2]), 'Depths', scene_folder_name[:-6],
                                          f'depth_map_{vid:04d}.pfm')
            proj_mat_filename = os.path.join("/".join(scene_path.split("/")[:-2]),
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4
            extrinsic[:3, 3] *= self.scale_factor
            intrinsics += [intrinsic.copy()]
            c2w = np.linalg.inv(extrinsic)
            c2ws += [c2w] 
            near_fars.append(near_far)
            rgb_files.append(img_filename)
            depth_files.append(depth_filename)

        intrinsics, c2ws, near_fars = [np.stack(element) for element in [intrinsics, c2ws, near_fars]]

        intrinsics_ = np.stack([np.eye(4)] * intrinsics.shape[0])
        intrinsics_[:,:3,:3] = intrinsics
        return intrinsics_, c2ws, near_fars, rgb_files, depth_files


    def add_single_scene(self, i, scene_path):
        intrinsics, c2w_mats, bds, rgb_files, depth_files = self.load_dtu_scene(scene_path)
        assert intrinsics.shape[0] == c2w_mats.shape[0] and c2w_mats.shape[0] == bds.shape[0] and bds.shape[0] == len(rgb_files) and len(rgb_files) == len(depth_files)
        near_depth = bds.min()
        far_depth = bds.max()
        # intrinsics, c2w_mats = batch_parse_llff_poses(poses, hw=[768,1024])

        i_test = self.get_i_test(c2w_mats.shape[0])
        i_train = self.get_i_train(c2w_mats.shape[0], i_test, self.mode)

        if self.mode is Mode.train:
            i_render = i_train
        else:
            i_render = i_test

        # Source images
        self.src_intrinsics.append(intrinsics[i_train])
        self.src_poses.append(c2w_mats[i_train])
        self.src_rgb_files.append([rgb_files[i] for i in i_train])
        self.src_depth_files.extend([depth_files[i] for i in i_train])

        # Target images
        num_render = len(i_render)
        self.render_rgb_files.extend([rgb_files[i] for i in i_render])
        self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
        self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])

        self.render_depth_files.extend([depth_files[i] for i in i_render])
        self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
        self.render_train_set_ids.extend([i] * num_render)
        self.render_ids.extend(i_render)


class DTUTestDataset(DTUDataset):
    name = 'dtu_test'
    dir_name = 'mvs_training/dtu/Rectified'
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


class DTUTrainDataset(DTUDataset):
    name = 'dtu'
    dir_name = 'mvs_training/dtu/Rectified'
    num_select_high = 3
    min_nearest_pose = 20

    def __len__(self):
        return len(self.render_rgb_files)

    # @staticmethod
    # def get_i_train(N, i_test, mode):
    #     if mode is Mode.train:
    #         return np.array(np.arange(N))
    #     else:
    #         return super().get_i_train(N, i_test, mode)




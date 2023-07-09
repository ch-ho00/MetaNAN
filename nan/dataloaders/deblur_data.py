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


class DeblurDataset(NoiseDataset, ABC):
    name = 'deblur'

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


    def get_i_test(self, N, holdout):
        return np.arange(N)[::holdout]

    def get_i_train(self, N, i_test):
        return np.array([j for j in np.arange(int(N)) if j not in i_test]) 

    @staticmethod
    def load_scene(scene_path, factor):
        return load_llff_data(scene_path, load_imgs=False, factor=factor)

    def __len__(self):
        return len(self.render_rgb_files) * 100000 if self.mode is Mode.train else len(self.render_rgb_files)

    def __getitem__(self, idx):
        return self.get_multiview_item(idx)

    def synfile2clean(self, rgb_file):
        noise_type = 'syn_motion' if 'synthetic_camera_motion_blur' in str(rgb_file) else 'syn_focus'
        folder_name = 'camera_motion_blur' if noise_type == 'syn_motion' else 'defocus_blur'
        noise_name = 'blur' if noise_type == 'syn_motion' else 'defocus'

        rgb_file = str(rgb_file).replace(str(self.folder_path) + '/', '')
        rgb_clean_file: Path = rgb_file.replace(folder_name, 'gt').replace(noise_name, 'gt').replace('images_1', 'raw').replace('images', 'raw')

        rgb_file = self.folder_path / rgb_file
        rgb_clean_file = self.folder_path / rgb_clean_file

        return rgb_file, rgb_clean_file


    def get_multiview_item(self, idx):
        # Read target data:
        eval_gain = 0 # self.args.eval_gain[idx // len(self.render_rgb_files)]
        idx = idx % len(self.render_rgb_files)
        rgb_file: Path = self.render_rgb_files[idx]
        
        blur_target = True
        scene_name = str(rgb_file).split('/')[-3]
        holdout = self.scene_holdout[scene_name]

        # image (H, W, 3)
        rgb = self.read_image(rgb_file)
        if self.mode == Mode.train:
            if 'synthetic' not in str(rgb_file) and idx % holdout == 0:
                blur_target = False            
            elif 'synthetic' in str(rgb_file) and (random.random() > 0.5 or self.mode != Mode.train):
                _, rgb_file_clean = self.synfile2clean(rgb_file)
                rgb = self.read_image(rgb_file_clean)
                blur_target = False            
        else:
            blur_target = False
            if 'synthetic' in str(rgb_file):
                _, rgb_file_clean = self.synfile2clean(rgb_file)
                rgb = self.read_image(rgb_file_clean)

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
            id_render = train_rgb_files.index(rgb_file)
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views # + np.random.randint(low=-2, high=self.num_select_high)
            id_render = id_render
        else:
            id_render = None
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = self.get_nearest_pose_ids(render_pose, depth_range, train_poses, subsample_factor, id_render)
        nearest_pose_ids = self.choose_views(nearest_pose_ids, num_select, id_render)
        assert None not in nearest_pose_ids
        src_rgbs = []
        src_cameras = []
        for src_id in nearest_pose_ids:
            src_rgb = self.read_image(train_rgb_files[src_id])
            src_rgbs.append(src_rgb)

            train_pose = train_poses[src_id]
            train_intrinsics_ = train_intrinsics[src_id]
            src_camera = self.create_camera_vector(src_rgb, train_intrinsics_, train_pose)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0) # (num_select, H, W, 3)
        src_cameras = np.stack(src_cameras, axis=0) # (num_select, 34)

        rgb, camera, src_rgbs, src_cameras = self.apply_transform(rgb, camera, src_rgbs, src_cameras)

        gt_depth = 0
        depth_range = self.final_depth_range(depth_range)
        return self.create_deblur_batch_from_numpy(rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range, gt_depth=gt_depth, eval_gain=eval_gain, blur_target=blur_target)

    def get_nearest_pose_ids(self, render_pose, depth_range, train_poses, subsample_factor, id_render):
        return get_nearest_pose_ids(render_pose,
                                    train_poses,
                                    min(self.num_source_views * subsample_factor, self.min_nearest_pose),
                                    tar_id=id_render,
                                    angular_dist_method='dist')

    def add_single_scene(self, i, scene_path, holdout, noise_type=None):
        _, poses, bds, render_poses, i_test, rgb_files = self.load_scene(scene_path, None if 'synthetic' not in str(scene_path) else 1)
        print(scene_path, len(poses))
        near_depth = bds.min()
        far_depth = bds.max()
        intrinsics, c2w_mats = batch_parse_llff_poses(poses, hw=[768,1024])
        if self.mode == Mode.train:
            i_render = list(range(poses.shape[0]))
            i_blurry = list(range(poses.shape[0]))
        else:
            i_render = self.get_i_test(poses.shape[0], holdout)
            i_blurry = self.get_i_train(poses.shape[0], i_render)

        # Source images
        self.src_intrinsics.append(intrinsics[i_blurry])
        self.src_poses.append(c2w_mats[i_blurry])
        self.src_rgb_files.append([rgb_files[i] for i in i_blurry])

        # Target images
        num_render = len(i_render)
        self.render_rgb_files.extend([rgb_files[i] for i in i_render])
        self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
        self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
        self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
        self.render_train_set_ids.extend([i] * num_render)
        self.render_ids.extend(i_render)


class DeblurTestDataset(DeblurDataset):
    name = 'deblur_test'
    dir_name = 'deblurnerf_dataset'
    num_select_high = 2
    min_nearest_pose = 28

    def apply_transform(self, rgb, camera, src_rgbs, src_cameras):
        if self.mode is Mode.train and self.random_crop:
            crop_h = np.random.randint(low=250, high=750) // 128 * 128
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(275 * 475 / crop_h // 128 * 128) #350 * 550
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                             (crop_h, crop_w))

        if self.mode is Mode.train and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        return rgb, camera, src_rgbs, src_cameras

    def final_depth_range(self, depth_range):
        return torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])


class DeblurTrainDataset(DeblurTestDataset):
    name = 'deblur'
    dir_name = 'deblurnerf_dataset'
    num_select_high = 3
    min_nearest_pose = 20

    def __len__(self):
        return len(self.render_rgb_files)





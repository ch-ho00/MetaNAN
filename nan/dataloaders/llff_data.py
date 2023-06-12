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
import random

from configs.local_setting import LOG_DIR
from nan.dataloaders.basic_dataset import NoiseDataset, re_linearize
from nan.dataloaders.data_utils import random_crop, get_nearest_pose_ids, random_flip, to_uint
from nan.dataloaders.llff_data_utils import load_llff_data, batch_parse_llff_poses
from nan.dataloaders.basic_dataset import Mode


class COLMAPDataset(NoiseDataset, ABC):
    name = 'colmap'

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

    def get_i_test(self, N):
        return np.arange(N)[::self.args.llffhold] if not self.args.degae_training else  np.array([j for j in np.arange(int(N))]) 

    def get_i_train(self, N, i_test, mode):
        return np.array([j for j in np.arange(int(N)) if j not in i_test]) if not self.args.degae_training else  np.array([j for j in np.arange(int(N))]) 

    @staticmethod
    def load_scene(scene_path, factor):
        return load_llff_data(scene_path, load_imgs=False, factor=factor)

    def __len__(self):
        if self.args.degae_training:
            return 100 if self.mode is Mode.train else len(self.render_rgb_files)
        else:
            return len(self.render_rgb_files) * 100000 if self.mode is Mode.train else len(self.render_rgb_files) * len(self.args.eval_gain)

    def __getitem__(self, idx):
        if self.args.degae_training:
            return self.get_singleview_item(idx)            
        else:
            return self.get_multiview_item(idx)

    def get_singleview_item(self, idx):
        # Read target data:
        eval_gain = self.args.eval_gain[idx // len(self.render_rgb_files)]
        idx = idx % len(self.render_rgb_files)
        rgb_file: Path = self.render_rgb_files[idx]
        # image (H, W, 3)
        rgb = self.read_image(rgb_file)

        # if self.mode is Mode.train:
        side = 512
        crop_h = np.random.randint(low=0, high=768 - side)
        crop_w =  np.random.randint(low=0, high=1024 - side)
        rgb = rgb[crop_h:crop_h+side, crop_w:crop_w+side]

        idx_ref = idx
        while idx == idx_ref:
            idx_ref = random.choice(list(range(len(self.render_rgb_files))))        
        rgb_file_ref: Path = self.render_rgb_files[idx_ref]
        # image (H, W, 3)
        rgb_ref = self.read_image(rgb_file_ref)

        # if self.mode is Mode.train:
        crop_h = np.random.randint(low=0, high=768 - side)
        crop_w =  np.random.randint(low=0, high=1024 -side)
        rgb_ref = rgb_ref[crop_h:crop_h+side, crop_w:crop_w+side]

        if self.mode in [Mode.train]: #, Mode.validation]:
            white_level = torch.clamp(10 ** -torch.rand(1), 0.6, 1)
        else:
            white_level = torch.Tensor([1])

        # d1
        rgb = re_linearize(torch.from_numpy(rgb[..., :3]), white_level)
        clean_d1 = False
        if self.mode is Mode.train:
            if random.random() > 0.25:
                rgb_d1 , _ = self.add_noise(rgb)
            else:
                rgb_d1 = rgb
                clean_d1 = True        
        else:
            rgb_d1, _ = self.add_noise_level(rgb, eval_gain)                        

        # d2
        rgb_ref = re_linearize(torch.from_numpy(rgb_ref[..., :3]), white_level)
        rgbs = torch.stack([rgb, rgb_ref])
        if self.mode is Mode.train:
            if random.random() > 0.25 or clean_d1:
                rgbs_d2, _ = self.add_noise(rgbs)        
            else:
                rgbs_d2 = rgbs
        else:
            rgbs_d2 = rgbs
        rgb_d2, rgb_ref_d2 = rgbs_d2[0], rgbs_d2[1]
        batch_dict = {'noisy_rgb' : rgb_d1.permute(2,0,1),
                      'clean_rgb' : rgb.permute(2,0,1),
                      'target_rgb' : rgb_d2.permute(2,0,1),
                      'ref_rgb' : rgb_ref_d2.permute(2,0,1),
                      'white_level'   : white_level}


        return batch_dict        

    def get_multiview_item(self, idx):
        # Read target data:
        eval_gain = self.args.eval_gain[idx // len(self.render_rgb_files)]
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
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=self.num_select_high)
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

        # Read the depth generated by IBRNet on the clean images
        try:
            scene = rgb_file.parent.parent.stem
            gt_depth_file = (LOG_DIR / "pretraining____clean__l2" / self.name / "same" / f"{scene}_255000")
            gt_depth_file = list(gt_depth_file.glob("*"))[0] / f"{rgb_file.stem}_depth_fine.png"
            gt_depth = imageio.imread(gt_depth_file).__array__() / 1000
        except (FileNotFoundError, IndexError):
            gt_depth = 0

        depth_range = self.final_depth_range(depth_range)
        return self.create_batch_from_numpy(rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range, gt_depth=gt_depth, eval_gain=eval_gain)

    def get_nearest_pose_ids(self, render_pose, depth_range, train_poses, subsample_factor, id_render):
        return get_nearest_pose_ids(render_pose,
                                    train_poses,
                                    min(self.num_source_views * subsample_factor, self.min_nearest_pose),
                                    tar_id=id_render,
                                    angular_dist_method='dist')

    def add_single_scene(self, i, scene_path):
        _, poses, bds, render_poses, i_test, rgb_files = self.load_scene(scene_path, self.args.factor)
        near_depth = bds.min()
        far_depth = bds.max()
        intrinsics, c2w_mats = batch_parse_llff_poses(poses, hw=[768,1024])

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


class LLFFTestDataset(COLMAPDataset):
    name = 'llff_test'
    dir_name = 'nerf_llff_data'
    num_select_high = 2
    min_nearest_pose = 28

    def apply_transform(self, rgb, camera, src_rgbs, src_cameras):
        if self.mode is Mode.train and self.random_crop:
            crop_h = np.random.randint(low=250, high=750) // 32 * 32
            # crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(375 * 575 / crop_h // 32 * 32)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                             (crop_h, crop_w))

        if self.mode is Mode.train and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        return rgb, camera, src_rgbs, src_cameras

    def final_depth_range(self, depth_range):
        return torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])


class LLFFDataset(LLFFTestDataset):
    name = 'llff'
    dir_name = 'real_iconic_noface'
    num_select_high = 3
    min_nearest_pose = 20

    def __len__(self):
        return len(self.render_rgb_files)

    @staticmethod
    def get_i_train(N, i_test, mode):
        if mode is Mode.train:
            return np.array(np.arange(N))
        else:
            return super().get_i_train(N, i_test, mode)




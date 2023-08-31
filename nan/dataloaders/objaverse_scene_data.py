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
import os, glob
import json
import imageio


class ObjaverseSceneDataset(NoiseDataset, ABC):
    name = 'objaverse'

    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.args = args 
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

    def get_i_test(self, N, holdout):
        return np.arange(N)[::holdout]

    def get_i_train(self, N, i_test):
        return np.array([j for j in np.arange(int(N)) if j not in i_test]) 

    @staticmethod
    def load_scene(folder, forward_facing=True):
        pose_file = folder / 'transforms.json'  # Update the file name to read from JSON
        
        # Load the JSON data
        with open(pose_file, 'r') as f:
            data = json.load(f)
        
        frames = data['frames']
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # Extract RGB files
        rgb_files = [os.path.join(folder, frame['file_path'].split(os.sep)[-1]) for frame in frames]

        sh = imageio.imread(rgb_files[0]).shape
        h, w = sh[:2]
        forward_facing = True if 'dimensions' not in data.keys() else False

        c2w_mats = []
        far = 0
        near = 100
        for frame in frames:
            matrix = np.array(frame['transform_matrix'])
            matrix = matrix @ blender2opencv
            matrix[:3, -1] = matrix[:3, -1]
            c2w_mats.append(matrix)
            if forward_facing:
                near    = 0.1
                far     = 0.6
            else:
                if far < np.linalg.norm(matrix[:3,-1]):
                    far = np.linalg.norm(matrix[:3,-1]) * 2

                if near > np.linalg.norm(matrix[:3,-1]) * 0.1:
                    near = np.linalg.norm(matrix[:3,-1]) * 0.1

        camera_angle_x = data['camera_angle_x']
        f = 0.5 * 800 / np.tan(0.5 * camera_angle_x)  # original focal length
        f *= w / 800  # modify focal length to match size self.img_wh

        intrinsics = np.array([[f, 0, w / 2., 0],
                               [0, f, h / 2., 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])


        intrinsics = [intrinsics for _ in frames]
        print("Dimension", data['dimensions'], " //  Radius", data['radius']) 
        print("Near Far : ",near, far)
        bds = [np.array([near, far]) for _ in frames]

        return np.array(c2w_mats), np.array(intrinsics), np.array(bds), rgb_files


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
        rgb_file_clean = str(rgb_file).split('/')
        rgb_file_clean[-1] = 'clean_' + rgb_file_clean[-1]
        rgb_file_clean = Path('/'.join(rgb_file_clean))
        scene_name = str(rgb_file).split('/')[-2]

        # image (H, W, 3)
        rgb = self.read_image(rgb_file_clean, multiple32=False, white_bkgd=self.args.white_bkgd)
        rgb_noisy = self.read_image(rgb_file, multiple32=False, white_bkgd=self.args.white_bkgd)
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

        nearest_pose_ids = self.get_nearest_pose_ids(render_pose, train_poses, subsample_factor, id_render)
        nearest_pose_ids = self.choose_views(nearest_pose_ids, num_select, id_render)
        # assert None not in nearest_pose_ids
        src_rgbs = []
        src_rgbs_clean = []
        src_cameras = []
        src_poses = []
        for src_id in nearest_pose_ids:
            if src_id is None:
                # print(self.render_rgb_files[idx])
                rgb_file = self.render_rgb_files[idx]
                src_rgb = self.read_image(rgb_file, multiple32=False, white_bkgd=self.args.white_bkgd)
                train_pose = self.render_poses[idx]
                train_intrinsics_ = self.render_intrinsics[idx]
            else:
                # print(train_rgb_files[src_id])
                rgb_file = train_rgb_files[src_id]
                src_rgb = self.read_image(rgb_file, multiple32=False, white_bkgd=self.args.white_bkgd)
                train_pose = train_poses[src_id]
                train_intrinsics_ = train_intrinsics[src_id]
                
            rgb_file_clean = str(rgb_file).split('/')
            rgb_file_clean[-1] = 'clean_' + rgb_file_clean[-1]
            rgb_file_clean = Path('/'.join(rgb_file_clean))
            src_rgb_clean = self.read_image(rgb_file_clean, multiple32=False, white_bkgd=self.args.white_bkgd)
            src_rgbs_clean.append(src_rgb_clean)
            src_poses.append(train_pose)
            src_rgbs.append(src_rgb)
            src_camera = self.create_camera_vector(src_rgb, train_intrinsics_, train_pose)
            src_cameras.append(src_camera)

        src_poses = np.stack(src_poses)

        src_rgbs = np.stack(src_rgbs, axis=0) # (num_select, H, W, 3)
        src_rgbs_clean = np.stack(src_rgbs_clean, axis=0)
        src_cameras = np.stack(src_cameras, axis=0) # (num_select, 34)

        rgb, camera, src_rgbs, src_cameras, rgb_noisy, src_rgbs_clean = self.apply_transform(rgb, camera, src_rgbs, src_cameras, rgb_noisy, src_rgbs_clean)

        gt_depth = 0
        depth_range = self.final_depth_range(depth_range)
        return self.create_objaverse_scene_batch_from_numpy(rgb, camera, rgb_file, src_rgbs, src_cameras, depth_range, gt_depth=gt_depth, eval_gain=eval_gain, rgb_noisy=rgb_noisy, src_rgbs_clean=src_rgbs_clean)

    def get_nearest_pose_ids(self, render_pose, train_poses, subsample_factor, id_render):
        return get_nearest_pose_ids(render_pose,
                                    train_poses,
                                    min(self.num_source_views * subsample_factor, self.min_nearest_pose),
                                    tar_id=id_render,
                                    angular_dist_method='dist')

    def add_single_scene(self, i, scene_path, holdout):
        c2w_mats, intrinsics, bds, rgb_files = self.load_scene(scene_path)
        near_depth = bds.min()
        far_depth = bds.max()
        i_test = self.get_i_test(c2w_mats.shape[0], holdout)
        i_blurry = self.get_i_train(c2w_mats.shape[0], i_test)
        i_render = i_blurry if self.mode == Mode.train else i_test
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


class ObjaverseSceneTestDataset(ObjaverseSceneDataset):
    name = 'objaverse_scene_test'
    dir_name = 'objaverse'
    num_select_high = 2
    min_nearest_pose = 28

    def apply_transform(self, rgb, camera, src_rgbs, src_cameras, rgb_noisy, src_rgbs_clean=None):
        if self.mode is Mode.train and self.random_crop:
            #crop_h = np.random.randint(low=250, high=750) // 128 * 128
            #crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            #crop_w = int(400 * 600 / crop_h // 128 * 128) #350 * 550
            #crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            crop_h = 400
            crop_w = 400
            rgb, camera, src_rgbs, src_cameras, rgb_noisy, src_rgbs_clean = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                             (crop_h, crop_w), rgb_noisy=rgb_noisy, src_rgbs_clean=src_rgbs_clean)

        if self.mode is Mode.train and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras, rgb_noisy, src_rgbs_clean = random_flip(rgb, camera, src_rgbs, src_cameras, rgb_noisy=rgb_noisy, src_rgbs_clean=src_rgbs_clean)

        return rgb, camera, src_rgbs, src_cameras, rgb_noisy, src_rgbs_clean

    def final_depth_range(self, depth_range):
        return torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])


class ObjaverseSceneTrainDataset(ObjaverseSceneTestDataset):
    name = 'objaverse_scene'
    dir_name = 'objaverse'
    num_select_high = 3
    min_nearest_pose = 20

    def __len__(self):
        return len(self.render_rgb_files)




from abc import ABC
from copy import copy
from pathlib import Path
import torch
from torch.utils.data import Dataset
from nan.dataloaders.data_utils import random_crop, random_flip
from configs.local_setting import DATA_DIR
import imageio
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from torchvision import transforms as T
from PIL import Image
import os, random, math
from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
import itertools
from pprint import pprint
import glob
import json 
from nan.dataloaders.objaverse_test_scenes import objaverse_test_scenes

class Mode(Enum):
    train = "train"
    validation = "validation"
    test = "test"


# From DeepRep
# https://github.com/goutamgmb/deep-rep/blob/master/data/postprocessing_functions.py#L73

t = 0.0031308
gamma = 2.4
a = 1. / (1. / (t ** (1 / gamma) * (1. - (1 / gamma))) - 1.)  # 0.055
# a = 0.055
k0 = (1 + a) * (1 / gamma) * t ** ((1 / gamma) - 1.)  # 12.92
# k0 = 12.92
inv_t = t * k0

transform = T.ToTensor()

def de_linearize(rgb, wl=1.):
    """
    Process the RGB values in the inverse process of the approximate linearization, in a differential format
    @param rgb:
    @param wl:
    @return:
    """
    completed = False
    if isinstance(wl, torch.Tensor):
        if wl.ndim > 1 and rgb.ndim == 4:
            assert wl.shape[0] == rgb.shape[0]
            completed = True
            srgb = []
            for signal, level in zip(rgb, wl):
                signal = signal / level
                srgb_ = torch.where(signal > t, (1 + a) * torch.clamp(signal, min=t) ** (1 / gamma) - a, k0 * signal)

                k1 = (1 + a) * (1 / gamma)
                srgb_ = torch.where(signal > 1, k1 * signal - k1 + 1, srgb_)
                srgb.append(srgb_)

            srgb = torch.stack(srgb)                
    if not completed:
        rgb = rgb / wl
        srgb = torch.where(rgb > t, (1 + a) * torch.clamp(rgb, min=t) ** (1 / gamma) - a, k0 * rgb)

        k1 = (1 + a) * (1 / gamma)
        srgb = torch.where(rgb > 1, k1 * rgb - k1 + 1, srgb)
    return srgb


def de_linearize_np(rgb, wl=1.):
    rgb = rgb / wl
    srgb = np.where(rgb > t, (1 + a) * np.clip(rgb, a_min=t, a_max=np.inf) ** (1 / gamma) - a, k0 * rgb)

    # From deep-rep/data/postprocessing_functions.py  DenoisingPostProcess
    k1 = (1 + a) * (1 / gamma)
    srgb = np.where(rgb > 1, k1 * rgb - k1 + 1, srgb)
    return srgb


def re_linearize(rgb, wl=1.):
    """
    Approximate re-linearization of RGB values by revert gamma correction and apply white level
    Revert gamma correction
    @param rgb:
    @param wl:
    @return:
    """
    # return rgb
    return wl * (rgb ** 2.2)
    # degamma = torch.where(rgb > inv_t, ((torch.clamp(rgb, min=inv_t) + a) / (1 + a)) ** gamma, rgb / k0)


class BurstDataset(Dataset, ABC):
    @property
    def dir_name(self) -> str:
        raise NotImplementedError

    @property
    def folder_path(self) -> Path:
        return DATA_DIR / self.dir_name

    def __init__(self, args, mode: Mode, scenes=(), random_crop=True):
        assert type(Mode.train) is Mode
        self.args = copy(args)
        self.mode = mode
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        s = 'Test ' if mode != Mode.train else ' Train'

        if self.args.train_dataset == 'objaverse' and mode == Mode.train:

            scene_root = os.path.join(DATA_DIR, self.dir_name)
            holdout = 8
            self.holdout = holdout            
            self.scenes_dirs = glob.glob(f'{scene_root}/final/objaverse_blur_dataset/*/*/blur_*')
            new_scene_dirs = []
            scene_dists = {}
            for i, folder in enumerate(self.scenes_dirs):
                if 'blur_mix' in folder:
                    continue
                if folder in objaverse_test_scenes:
                    continue
                folder_path = Path(folder)
                pose_file = folder_path / 'transforms.json'  # Update the file name to read from JSON
                if os.path.exists(pose_file):
                    with open(pose_file, 'r') as f:
                        data = json.load(f)
                    near = data['near'] * 0.8
                    far = data['far'] * 1.2
                    scale = 1 / (near * 0.75)
                    far *= scale
                    # print(folder, far)
                    new_scene_dirs.append(folder)
                    # if far > 3.5:
                    #     new_scene_dirs.append(folder)
                    #     scene_dists[folder] = far
                else:
                    print("Pose file missing", pose_file)
            self.scenes_dirs = new_scene_dirs

            for i, scene_path in enumerate(self.scenes_dirs): 
                self.add_single_scene(i, scene_path, holdout)


        elif self.args.eval_dataset == 'objaverse_test':
            holdout = 8
            self.scenes_dirs = objaverse_test_scenes
            for i, scene_path in enumerate(self.scenes_dirs): 
                print(scene_path)
                self.add_single_scene(i, scene_path, holdout)

        elif self.args.train_dataset == 'objaverse_scene':
            assert isinstance(args.train_scenes, list)
            assert len(args.train_scenes) == 1
            scene_root = os.path.join(DATA_DIR, self.dir_name, self.args.train_scenes[0])
            holdout = 16
            self.holdout = holdout
            self.add_single_scene(0, Path(scene_root), holdout)

        elif self.args.train_dataset == 'deblur_scene':
            assert isinstance(args.train_scenes, list)
            assert len(args.train_scenes) == 1
            scene_root = os.path.join(DATA_DIR, self.dir_name, self.args.train_scenes[0])
            holdout = 8
            self.holdout = holdout
            self.add_single_scene(0, Path(scene_root), holdout)

        elif self.args.train_dataset == 'deblur' or self.args.train_dataset == 'objaverse':

            self.scenes = ['blurcozy2room',  'blurpool' ,  'blurwine',  'roomblur_low', 'blurfactory'  ,  'blurtanabata',  'dark'    ,  'roomblur_high']
            self.scenes.sort()
            if self.args.train_dataset == 'deblur':
                self.scenes = self.args.eval_scenes if mode != Mode.train else [scene for scene in self.scenes if scene not in self.args.eval_scenes]
            else:
                self.scenes = ['blurcozy2room',  'blurpool' ,  'blurwine',  'roomblur_low', 'blurfactory'  ,  'blurtanabata',  'dark'    ,  'roomblur_high']
                # self.scenes = ['blurfactory', 'blurcozy2room', 'blurpool', 'blurtanabata'] 
            
            print(f"############ Loading {s} Dataset #############")
            self.scenes_dirs = []
            for cnt, scene in enumerate(self.scenes):
                data_root = os.path.join(DATA_DIR, self.dir_name)
                scene_root = os.path.join(data_root, scene)
                self.scenes_dirs.append(scene_root)
                self.add_single_scene(cnt, Path(scene_root), holdout=8)

        else:
            self.scenes_dirs = self.pick_scenes(scenes)
            if len(self.scenes_dirs) == 1:
                print(f"loading {self.scenes_dirs[0].stem} scenes for {mode}")
            else:
                print(f"loading {len(self.scenes_dirs)} scenes for {mode}")
            print(f"num of source views {self.num_source_views}")

            self.scenes_dirs = [dir_ for dir_ in self.scenes_dirs if 'sparse' not in str(dir_)]
            # if self.mode != Mode.train:
            #     self.scenes_dirs.sort()
            #     pprint(self.scenes_dirs)

            for i, scene_path in enumerate(self.scenes_dirs):
                self.add_single_scene(i, scene_path)

        print(f"Loaded {s} Img File = {len(self.render_rgb_files)} from {len(self.scenes_dirs)} scenes")

    def pick_scenes(self, scenes):
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
            return self.get_specific_scenes(scenes)
        else:
            return self.get_all_scenes()

    def get_specific_scenes(self, scenes):
        return [self.folder_path / scene for scene in scenes]

    def get_all_scenes(self):
        return self.listdir(self.folder_path)

    @staticmethod
    def listdir(folder):
        return list(folder.glob("*"))

    # @staticmethod
    # def read_image(filename, **kwargs):
    #     return imageio.imread(filename).astype(np.float32) / 255.

    @staticmethod
    def read_image(filename, multiple32=True, img_wh=None, white_bkgd=False, **kwargs):
        
        if white_bkgd:
            img = Image.open(filename).convert('RGBA')        
        else:
            img = Image.open(filename).convert('RGB')

        if multiple32 and img_wh == None:
            img = img.resize([1024, 768], Image.LANCZOS)
        elif img_wh:
            img = img.resize(img_wh, Image.LANCZOS)

        img = transform(img)
        if white_bkgd:
            alpha = img[-1:]
            img = img[:3] * alpha + (1 - alpha)
            return img.permute(1,2,0).numpy(), alpha.permute(1,2,0).numpy()

        return img.permute(1,2,0).numpy()

    def apply_transform(self, rgb, camera, src_rgbs, src_cameras):
        if self.mode == Mode.train and self.random_crop:
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras)

        if self.mode == Mode.train and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        return rgb, camera, src_rgbs, src_cameras

    def final_depth_range(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def create_camera_vector(rgb, intrinsics, pose):
        """
        Creating camera representative vector (used by IBRNet)
        :param rgb: (H, W, 3)
        :param intrinsics: (4, 4)
        :param pose: (4, 4)
        :return: camera vector (34) (H, W, K.flatten(), (R|t).flatten())
        """
        return np.concatenate((rgb.shape[:2], intrinsics.flatten(), pose.flatten())).astype(np.float32)


class NoiseDataset(BurstDataset, ABC):
    def __init__(self, args, mode, **kwargs):
        super().__init__(args, mode, **kwargs)
        if mode == Mode.train:
            assert len(self.args.std) == 4
            self.get_noise_params = self.get_noise_params_train
        else:
            if self.args.eval_gain == 0:
                sig_read, sig_shot = 0, 0
                print(f"Loading {mode} set without additional noise.")
            else:
                # load gain data from KPN paper https://bmild.github.io/kpn/index.html
                noise_data = np.load(DATA_DIR / 'synthetic_5d_j2_16_noiselevels6_wide_438x202x320x8.npz')
                sig_read_list = np.unique(noise_data['sig_read'])[2:]
                sig_shot_list = np.unique(noise_data['sig_shot'])[2:]

                self.log_sig_read = np.log10(sig_read_list)
                self.log_sig_shot = np.log10(sig_shot_list)

                self.d_read = np.diff(self.log_sig_read)[0]
                self.d_shot = np.diff(self.log_sig_shot)[0]

        self.depth_range = None

    def choose_views(self, possible_views, num_views, target_view):
        if self.mode == Mode.train:
            chosen_ids = np.random.choice(possible_views, min(num_views, len(possible_views)), replace=False).tolist()
        else:
            chosen_ids = possible_views[:min(num_views, len(possible_views))]

        assert target_view not in chosen_ids

        if self.args.include_target:
            # always include input image in the first idx - denoising task
            chosen_ids[0] = target_view

        return chosen_ids

    def get_noise_params_train(self):
        sigma_read_lim = self.args.std[:2]
        sigma_shot_lim = self.args.std[2:]

        sigma_read_log = np.random.default_rng().uniform(low=sigma_read_lim[0], high=sigma_read_lim[1], size=1).item()
        sigma_shot_log = np.random.default_rng().uniform(low=sigma_shot_lim[0], high=sigma_shot_lim[1], size=1).item()

        sigma_read = 10 ** sigma_read_log
        sigma_shot = 10 ** sigma_shot_log

        return sigma_read, sigma_shot

    def get_noise_params_level(self, gain_level):

        gain_read = gain_shot = np.log2(gain_level)
        sig_read = 10 ** (self.log_sig_read[0] + self.d_read * gain_read)
        sig_shot = 10 ** (self.log_sig_shot[0] + self.d_shot * gain_shot)

        return sig_read, sig_shot

    def add_noise_level(self, rgb, gain_level):
        sig_read, sig_shot = self.get_noise_params_level(gain_level)
        std = self.get_std(rgb, sig_read, sig_shot)
        noise = std * torch.randn_like(rgb)

        noise_rgb = rgb + noise
        sigma_estimate = self.get_std(noise_rgb.clamp(0, 1), sig_read, sig_shot)
        return noise_rgb, sigma_estimate

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
    

    @classmethod
    def get_std(cls, rgb, sig_read, sig_shot):
        return (sig_read ** 2 + sig_shot ** 2 * rgb) ** 0.5

    def add_noise(self, rgb):
        sig_read, sig_shot = self.get_noise_params()
        std = self.get_std(rgb, sig_read, sig_shot)
        noise = std * torch.randn_like(rgb)
        noise_rgb = rgb + noise
        sigma_estimate = self.get_std(noise_rgb.clamp(0, 1), sig_read, sig_shot)
        return noise_rgb, sigma_estimate

    def create_batch_from_numpy(self, rgb_clean, camera, rgb_file, src_rgbs_clean, src_cameras, depth_range,
                                gt_depth=None, eval_gain=1):
        if self.mode in [Mode.train]: #, Mode.validation]:
            white_level = (10 ** -torch.rand(1)) * 0.4 + 0.6
        else:
            white_level = torch.Tensor([1])

        if rgb_clean is not None:
            rgb_clean = re_linearize(torch.from_numpy(rgb_clean[..., :3]), white_level)
            if self.mode is Mode.train:
                rgb, _ = self.add_noise(rgb_clean)        
            else:
                rgb, _ = self.add_noise_level(rgb_clean, eval_gain)                        
        else:
            rgb = None
        src_rgbs_clean = re_linearize(torch.from_numpy(src_rgbs_clean[..., :3]), white_level)
        if self.mode is Mode.train:
            src_rgbs, sigma_est = self.add_noise(src_rgbs_clean)
        else:
            src_rgbs, sigma_est = self.add_noise_level(src_rgbs_clean, eval_gain)
                      
        batch_dict = {'camera'        : torch.from_numpy(camera),
                      'rgb_path'      : str(rgb_file),
                      'src_rgbs_clean': src_rgbs_clean,
                      'src_rgbs'      : src_rgbs,
                      'src_cameras'   : torch.from_numpy(src_cameras),
                      'depth_range'   : depth_range,
                      'sigma_estimate': sigma_est,
                      'white_level'   : white_level,
                      'eval_gain' : eval_gain}


        if rgb_clean is not None:
            batch_dict['rgb_clean'] = rgb_clean
            batch_dict['rgb'] = rgb

        if gt_depth is not None:
            batch_dict['gt_depth'] = gt_depth

        return batch_dict

    def create_objaverse_scene_batch_from_numpy(self, rgb_clean, camera, rgb_file, src_rgbs, src_cameras, depth_range,
                                gt_depth=None, eval_gain=1, rgb_noisy=None, src_rgbs_clean=None, alpha_clean=None):

        if self.mode in [Mode.train]:
            white_level = (10 ** -torch.rand(1)) * 0.5 + 0.5
        else:
            white_level = torch.Tensor([1])

        if self.args.add_burst_noise:
            rgb_clean = re_linearize(torch.from_numpy(rgb_clean[..., :3]), white_level)
            rgb_noisy = re_linearize(torch.from_numpy(rgb_noisy[..., :3]), white_level)
            if self.mode is Mode.train:
                rgb, _ = self.add_noise(rgb_clean)        
            else:
                rgb, _ = self.add_noise_level(rgb_clean, eval_gain)                        

            src_rgbs = re_linearize(torch.from_numpy(src_rgbs[..., :3]), white_level)
            src_rgbs_clean = re_linearize(torch.from_numpy(src_rgbs_clean[..., :3]), white_level)
            if self.mode is Mode.train:
                src_rgbs, sigma_est = self.add_noise(src_rgbs)
            else:
                src_rgbs, sigma_est = self.add_noise_level(src_rgbs, eval_gain)

            batch_dict = {
                'white_level' : white_level,
                'sigma_estimate' : sigma_est
            }
        else:
            if rgb_clean is not None:
                rgb_clean = torch.from_numpy(rgb_clean[..., :3])
            else:
                rgb = None
            src_rgbs        = torch.from_numpy(src_rgbs[..., :3])
            src_rgbs_clean  = torch.from_numpy(src_rgbs_clean[..., :3])
            eval_gain = 0
            batch_dict = {}

        batch_dict.update({
                      'camera'         : torch.from_numpy(camera),
                      'rgb_path'       : str(rgb_file),
                      'src_rgbs'       : src_rgbs,
                      'src_rgbs_clean' : src_rgbs_clean,
                      'src_cameras'    : torch.from_numpy(src_cameras),
                      'depth_range'    : depth_range,
                      'eval_gain'      : eval_gain,
                      'rgb_clean'      : rgb_clean,
                      'rgb'            : rgb_noisy,
                      'alpha_clean'    : alpha_clean
                      })

        return batch_dict



if __name__ == '__main__':
    v = torch.linspace(0, 1, 100)
    v_unproc = re_linearize(v, 1)
    v_proc = de_linearize(v_unproc, 1)

    plt.figure()
    plt.plot(v, v, label='linear')
    plt.plot(v, v_unproc, label='degamma')
    plt.plot(v, v_proc, label='gamma')
    plt.legend()
    plt.show()

    im_path = DATA_DIR / 'nerf_llff_data' / 'fern' / 'images_4' / 'image000.png'
    im = torch.from_numpy(imageio.imread(im_path) / 255)
    im = im + torch.randn_like(im) * 0.1
    print(im_path)

    plt.figure()
    plt.imshow(im.clamp(0, 1))
    plt.show()

    im_unprocessed = re_linearize(im)

    plt.figure()
    plt.imshow(im_unprocessed.clamp(0, 1))
    plt.show()

    im_processes = de_linearize(im_unprocessed)

    plt.figure()
    plt.imshow(im_processes.clamp(0, 1))
    plt.show()
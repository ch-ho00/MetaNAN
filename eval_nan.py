import random 
import sys, os
from tqdm import tqdm
import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path

import numpy as np
import torch

from configs.local_setting import EVAL_CONFIG
from configs.config import CustomArgumentParser

from nan.dataloaders.create_training_dataset import create_training_dataset
from nan.dataloaders.basic_dataset import Mode, de_linearize
from nan.dataloaders import dataset_dict
from nan.model import NANScheme
from nan.sample_ray import RaySampler
from nan.render_image import render_single_image
from nan.utils.general_utils import img_HWC2CHW
from nan.utils.io_utils import print_link, colorize
from nan.utils.eval_utils import mse2psnr, img2psnr
import pickle
from pprint import pprint
import lpips
import pytorch_msssim


sys.argv = sys.argv + ['--config', str(EVAL_CONFIG)]
# Create training args
parser = CustomArgumentParser.config_parser()
args = parser.parse_args(verbose=True)
device = torch.device(f"cuda:{args.local_rank}")
lpips_loss = lpips.LPIPS(net='vgg') #.to(device)

for key, value in sorted(vars(args).items()):
    print(f"{key:<30}: {value}")

# Set distributed options
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()

model_name = str(args.ckpt_path).split('/')[-1][:-4]
save_folder = Path(f'eval_result/{model_name}')
save_folder.mkdir(parents=True, exist_ok=True)
skip_forward = False

args.eval_gain = [1,2,4,8,16,20] # 
if not skip_forward:
    val_dataset = dataset_dict["llff_test"](args, Mode.validation, scenes=[])
    print("###################################")
    print(' Val Dataset Size = ', len(val_dataset))
    print("#################################")

    load_ckpt = args.ckpt_path
    model = NANScheme.create(args)
    checkpoint = torch.load(load_ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])

    N_val_imgs = len(val_dataset.render_rgb_files)
    render_result = {}
    visualize_freq = 2

    for idx in tqdm(range(N_val_imgs), desc='Validation Image Number'):
        render_result[idx] = {}
        for lvl_cnt, gain_level in enumerate(args.eval_gain): 

            val_idx = N_val_imgs * lvl_cnt + idx
            val_data = val_dataset[val_idx]
            val_data = {k : val_data[k][None] if isinstance(val_data[k], torch.Tensor) else val_data[k] for k in val_data.keys()}
            ray_sampler = RaySampler(val_data, device, render_stride=args.render_stride)
            H, W = ray_sampler.H, ray_sampler.W
            gt_img = ray_sampler.rgb_clean.reshape(H, W, 3)

            model.switch_to_eval()
            with torch.no_grad():
                ret = render_single_image(ray_sampler=ray_sampler, model=model, args=args, eval_=True)
                    
            rgb_gt = img_HWC2CHW(gt_img)[...,::args.render_stride, ::args.render_stride]
            if idx % visualize_freq == 0:
                fn = f'{model_name}_gain{gain_level}_{idx}.png'
                save_dir = os.path.join(str(save_folder), fn) 

                sample_src = ray_sampler.src_rgbs.cpu()[0,0, ::args.render_stride, ::args.render_stride].permute(2,0,1)
                rgb_fine = img_HWC2CHW(ret['fine'].rgb.detach().cpu())
                rgb_im = torch.cat((sample_src, rgb_gt, rgb_fine), dim=-1)
                rgb_im = de_linearize(rgb_im, ray_sampler.white_level).clamp(min=0., max=1.)
                depth_im = img_HWC2CHW(colorize(ret['fine'].depth.detach().cpu(), cmap_name='jet', append_cbar=True))
                rgb_im = torch.cat((rgb_im, depth_im), dim=-1)


                if args.bpn_prenet:
                    h, w, _ = ret['bpn_reconst'].shape[-3:]
                    reconst_img = ret['bpn_reconst'][0,0].permute(2,0,1)[...,::args.render_stride, ::args.render_stride]
                    reconst_img = de_linearize(reconst_img.cpu(), ray_sampler.white_level).clamp(0,1)
                    rgb_im = torch.cat((rgb_im, reconst_img.cpu()), dim=-1)

                plt.imsave(save_dir, rgb_im.permute(1,2,0).cpu().numpy())

                del depth_im 
            # write scalar
            pred_rgb = ret['fine'].rgb

            pred_rgb_tensor = de_linearize(ret['fine'].rgb, ray_sampler.white_level).clamp(0,1).permute(2,0,1)[None]
            gt_rgb_tensor = de_linearize(gt_img[::args.render_stride, ::args.render_stride], ray_sampler.white_level).clamp(0,1).permute(2,0,1)[None]
            # Calculate LPIPS
            lpips_value = lpips_loss(pred_rgb_tensor, gt_rgb_tensor).item()
            # Calculate SSIM
            ssim_value = pytorch_msssim.ssim(pred_rgb_tensor, gt_rgb_tensor, data_range=1, size_average=True).item()
            psnr_curr_img = img2psnr(pred_rgb_tensor, gt_rgb_tensor)
            print(gain_level, psnr_curr_img, ssim_value, lpips_value)

            render_result[idx][gain_level] = {
                'psnr': psnr_curr_img,
                'lpips': lpips_value,
                'ssim': ssim_value,
                'depth': ret['fine'].depth.detach().cpu(),
                'rgb': pred_rgb_tensor
            }

        with open(f'./{str(save_folder)}/result.pkl', 'wb') as fp:
            pickle.dump(render_result, fp)


### 
with open(f'./{str(save_folder)}/result.pkl', 'rb') as fp:
    render_result = pickle.load(fp)

final_result = {}
results_level = {k : [] for k in args.eval_gain}
results_lpips = {k : [] for k in args.eval_gain}
results_ssim = {k : [] for k in args.eval_gain}

for img_idx in render_result.keys():
    for gain_level in render_result[img_idx].keys():
        results_level[gain_level].append(render_result[img_idx][gain_level]['psnr'])
        results_level[gain_level].append(render_result[img_idx][gain_level]['psnr'])
        results_lpips[gain_level].append(render_result[img_idx][gain_level]['lpips'])
        results_ssim[gain_level].append(render_result[img_idx][gain_level]['ssim'])


for level in args.eval_gain:
    results_level[str(level) + "_avg"] = np.mean(results_level[level])
    results_level[str(level) + "_avg_lpips"] = np.mean(results_lpips[level])
    results_level[str(level) + "_avg_ssim"] = np.mean(results_ssim[level])

final_result.update(results_level)
pprint(results_level)

depth_std = {}
rgb_std = {}
for img_idx in render_result.keys():
    pred_rgbs = np.stack([render_result[img_idx][gain_level]['rgb'] for gain_level in render_result[img_idx].keys()])
    pred_depths = np.stack([render_result[img_idx][gain_level]['depth'] for gain_level in render_result[img_idx].keys()])
    rgb_std[img_idx] = np.std(pred_rgbs, axis=0).mean()
    depth_std[img_idx] = np.std(pred_depths, axis=0).mean()


# percentile = 15

# rgb_std   = np.array(list(rgb_std.values()))
# rgb_top = np.percentile(rgb_std , percentile)
# rgb_bottom = np.percentile(rgb_std , 100 - percentile)
# final_result['mean_rgb_std']   = np.std(rgb_std[(rgb_std >= rgb_top) & (rgb_std <= rgb_bottom)])

# depth_std   = np.array(list(depth_std.values()))
# depth_top = np.percentile(depth_std , percentile)
# depth_bottom = np.percentile(depth_std , 100 - percentile)
# final_result['mean_depth_std']   = np.std(depth_std[(depth_std >= depth_top) & (depth_std <= depth_bottom)])

with open(f'./{str(save_folder)}/result_final.pkl', 'wb') as fp:
    pickle.dump(final_result, fp)

import pdb; pdb.set_trace()
pprint(final_result)


# depth_std = {}
# rgb_std = {}
# for img_idx in render_result.keys():
#     pred_rgbs = np.stack([render_result[img_idx][gain_level]['rgb'] for gain_level in render_result[img_idx].keys()])
#     pred_depths = np.stack([render_result[img_idx][gain_level]['depth'] for gain_level in render_result[img_idx].keys()])
#     rgb_std[img_idx] = np.std(pred_rgbs, axis=0).mean()
#     depth_std[img_idx] = np.std(pred_depths, axis=0).mean()

# final_result['mean_rgb_std'] = np.mean(list(rgb_std.values()))
# final_result['mean_depth_std'] = np.mean(list(depth_std.values()))

# with open(f'./{str(save_folder)}/result_final.pkl', 'wb') as fp:
#     pickle.dump(final_result, fp)

# pprint(final_result)

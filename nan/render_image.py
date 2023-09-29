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

from typing import Dict
import torch
from collections import OrderedDict

from tqdm import tqdm

from nan.render_ray import RayRender
from nan.raw2output import RaysOutput
from nan.sample_ray import RaySampler
from nan.se3 import SE3_to_se3_N, get_spline_poses
from nan.projection import warp_latent_imgs
from nan.dataloaders.data_utils import get_nearest_pose_ids
import torch.nn.functional as F

alpha=0.9998

def render_single_image(ray_sampler: RaySampler,
                        model,
                        args,
                        save_pixel=None,
                        global_step=0,
                        eval_=False) -> Dict[str, RaysOutput]:
    """
    :param: save_pixel:
    :param: featmaps:
    :param: render_stride:
    :param: white_bkgd:
    :param: det:
    :param: ret_output:
    :param: projector:
    :param: ray_batch:
    :param: ray_sampler: RaySamplingSingleImage for this view
    :param: model:  {'net_coarse': , 'net_fine': , ...}
    :param: chunk_size: number of rays in a chunk
    :param: N_samples: samples along each ray (for both coarse and fine model)
    :param: inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param: N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :return: {'coarse': {'rgb': numpy, 'depth': numpy, ...}, 'fine': {}}
    """
    if eval_:
        w = 0
    else:
        w = alpha ** global_step

    device = torch.device(f'cuda:{args.local_rank}')
    ray_render = RayRender(model=model, args=args, device=device, save_pixel=save_pixel)
    if args.clean_src_imgs:
        org_src_rgbs = ray_sampler.src_rgbs_clean.to(device)
    else:
        org_src_rgbs = ray_sampler.src_rgbs.to(device)
    sigma_est = ray_sampler.sigma_estimate.to(device) if ray_sampler.sigma_estimate != None else None
    src_cameras = ray_sampler.src_cameras.to(device)

    if args.burst_length > 1:
        nearby_idxs = []
        poses = src_cameras[:,:,-16:].reshape(-1, 4, 4).cpu().numpy()
        for pose in poses:
            ids = get_nearest_pose_ids(pose, poses, args.burst_length, angular_dist_method='dist', sort_by_dist=True)
            nearby_idxs.append(ids)
    else:
        nearby_idxs = None


    src_rgbs, featmaps = ray_render.calc_featmaps(org_src_rgbs, white_level=ray_sampler.white_level, weight=w, nearby_idxs=nearby_idxs, src_cameras=src_cameras)
    all_ret = OrderedDict([('coarse', RaysOutput.empty_ret()),
                           ('fine', None)])

    H, W = org_src_rgbs.shape[-3:-1]

    # if 'pred_kernel' in featmaps.keys():
    #     kernel_size = featmaps['pred_kernel'].shape[1]
    #     clean_src_imgs = ray_sampler.src_rgbs_clean.to(device)[0].permute(0,3,1,2)
    #     clean_src_imgs = F.pad(clean_src_imgs, [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2])
    #     unfolded = F.unfold(clean_src_imgs, kernel_size)
    #     img_stack = unfolded.reshape(args.num_source_views, clean_src_imgs.shape[1], kernel_size, kernel_size, H, W)
    #     pred_noisy = (img_stack * featmaps['pred_kernel'][:,None]).sum(2).sum(2)            
    #     all_ret['kernel_reconst'] = pred_noisy

    if 'reconst_img' in featmaps.keys():
        all_ret['kernel_reconst'] = featmaps['reconst_img']
        
    if 'pred_offset' in featmaps.keys():
        all_ret['bpn_reconst'] = src_rgbs
        num_latent = 4
        src_poses = src_cameras[:,:,-16:].reshape(-1, 4, 4)[:,:3,:4]
        src_se3_start = SE3_to_se3_N(src_poses)                                                                                                     # (n_src, 6)             
        src_se3_end = src_se3_start + featmaps['pred_offset']                                                                                       # (n_src, 6)             
        src_spline_poses = get_spline_poses(src_se3_start, src_se3_end, spline_num=num_latent)                                                      # (n_src, n_latent, 3, 4) 

        src_spline_poses_4x4 =  torch.eye(4)[None,None].expand(args.num_source_views, num_latent, 4,4)
        src_spline_poses_4x4 = src_spline_poses_4x4.to(src_spline_poses.device)
        src_spline_poses_4x4[:,:, :3, :4] = src_spline_poses

        # repeat source images
        org_src_rgbs_ = org_src_rgbs if 'reconst_img' not in featmaps.keys() else featmaps['reconst_img'].permute(0,2,3,1)[None]
        org_src_rgbs_ = org_src_rgbs_.unsqueeze(2)
        org_src_rgbs_ = org_src_rgbs_.expand(1, args.num_source_views, num_latent, H, W, 3)                           # (1, n_src, n_latent, H, W, 3)
        org_src_rgbs_ = org_src_rgbs_.reshape(1, -1, H, W ,3)

        src_rgbs        = src_rgbs[:,:, None].expand(1, args.num_source_views, num_latent, H, W, 3)
        src_rgbs        = src_rgbs.reshape(1, -1, H, W ,3)

        for level in ['coarse', 'fine']:
            sh =  featmaps[level].shape[-2:]
            featmaps[level] = featmaps[level][:, None].expand(args.num_source_views, num_latent, args.fine_feat_dim, sh[0], sh[1])
            featmaps[level] = featmaps[level].reshape(-1, args.fine_feat_dim, sh[0], sh[1])

        print("OFFSET = ", featmaps['pred_offset'])

    else:
        org_src_rgbs_ = org_src_rgbs if 'reconst_img' not in featmaps.keys() else featmaps['reconst_img'].permute(0,2,3,1)[None]

    if args.N_importance > 0:
        all_ret['fine'] = RaysOutput.empty_ret()
    N_rays = ray_sampler.rays_o.shape[0]

    H, W = org_src_rgbs.shape[-3:-1]
    for i in tqdm(range(0, N_rays, args.chunk_size)):
        # print('batch', i)
        ray_batch = ray_sampler.specific_ray_batch(slice(i, i + args.chunk_size, 1), clean=args.sup_clean)
        if 'pred_offset' in featmaps.keys():
            # Attach intrinsics and HW vector
            intrinsics = ray_batch['src_cameras'][:,:,2:18].reshape(-1, 4, 4)                                                                           # (n_src, 4, 4)            
            src_latent_camera = ray_batch['src_cameras'][:,:,:-16][:,:, None].expand(1,args.num_source_views, num_latent, 18)
            src_latent_camera = torch.cat([src_latent_camera, src_spline_poses_4x4.reshape(1, args.num_source_views, num_latent, -1)], dim=-1)     # (1, n_src, n_latent, 34)
            src_latent_camera[:,:,0] = ray_batch['src_cameras']
            ray_batch['src_cameras'] = src_latent_camera.reshape(1,-1,34)
            ray_batch['pred_offset'] = featmaps['pred_offset']

        ret       = ray_render.render_batch(ray_batch=ray_batch,
                                            proc_src_rgbs=src_rgbs,
                                            featmaps=featmaps,
                                            org_src_rgbs=org_src_rgbs_,
                                            sigma_estimate=sigma_est)
        all_ret['coarse'].append(ret['coarse'])
        if ret['fine'] is not None:
            all_ret['fine'].append(ret['fine'])
        torch.cuda.empty_cache()
        del ret
        
    # merge chunk results and reshape
    out_shape = torch.empty(ray_sampler.H, ray_sampler.W)[::args.render_stride, ::args.render_stride].shape
    all_ret['coarse'].merge(out_shape)
    if all_ret['fine'] is not None:
        all_ret['fine'].merge(out_shape)

    return all_ret




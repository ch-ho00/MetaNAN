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
from nan.dataloaders.data_utils import get_nearest_pose_ids, get_depth_warp_img
import torch.nn.functional as F

alpha=0.9998

def render_single_image(ray_sampler: RaySampler,
                        model,
                        args,
                        save_pixel=None,
                        global_step=0,
                        eval_=False,
                        clean_src_imgs=False,
                        denoised_input=None) -> Dict[str, RaysOutput]:
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

    all_ret = OrderedDict([('coarse', RaysOutput.empty_ret()),
                           ('fine', None)])
    device = torch.device(f'cuda:{args.local_rank}')
    ray_render = RayRender(model=model, args=args, device=device, save_pixel=save_pixel)
    if denoised_input:
        input_src_rgbs = ray_sampler.denoised_src_rgbs.to(device)
    elif args.clean_src_imgs or clean_src_imgs:
        input_src_rgbs = ray_sampler.src_rgbs_clean.to(device)
    else:
        input_src_rgbs = ray_sampler.src_rgbs.to(device)

    H, W = input_src_rgbs.shape[-3:-1]
    sigma_est = ray_sampler.sigma_estimate.to(device) if ray_sampler.sigma_estimate != None else None
    src_cameras = ray_sampler.src_cameras.to(device)

    if args.burst_length > 1:
        ray_batch = ray_sampler.specific_ray_batch(slice(0, args.chunk_size, 1), clean=args.sup_clean)
        nearby_idxs = []
        src_poses = ray_batch['src_cameras'][:,:,-16:].reshape(-1, 4, 4).cpu().numpy()
        src_intrinsics = ray_batch['src_cameras'][0, :,2:18].reshape(-1, 4, 4)
        for pose in src_poses:
            ids = get_nearest_pose_ids(pose, src_poses, args.burst_length, angular_dist_method='dist', sort_by_dist=True)
            nearby_idxs.append(ids)

        nearby_imgs = torch.stack([input_src_rgbs[0].permute(0,3,1,2)[ids] for ids in nearby_idxs])
        nearby_poses = torch.stack([torch.from_numpy(src_poses[ids]) for ids in nearby_idxs]).to(device)
        nearby_intrinsics = torch.stack([src_intrinsics[ids] for ids in nearby_idxs])[..., :3,:3]

        extrinsics = torch.inverse(nearby_poses)
        input_imgs = [img for img in nearby_imgs.permute(1,0,2,3,4)]
        depth, _, _ = model.patchmatch(input_imgs, nearby_intrinsics, extrinsics, ray_batch['depth_range'][0,0].repeat(args.num_source_views), ray_batch['depth_range'][0,1].repeat(args.num_source_views))            

        src_rgbd = torch.cat([input_src_rgbs[0].permute(0,3,1,2), depth], dim=1)
        warped_rgbds, coords = get_depth_warp_img(nearby_imgs, nearby_poses, src_intrinsics, depth, nearby_idxs)
        print(round((((coords[..., 0] > 0) & (coords[..., 0] < W)) / coords[..., 0].numel()).sum().item(), 3), round((((coords[..., 1] > 0) & (coords[..., 1] < H)) / coords[..., 1].numel()).sum().item(), 3))

        ref_rgbd = torch.cat([nearby_imgs[:, :1], depth.unsqueeze(1)], dim=2)
        input_imgs = torch.cat([ref_rgbd, warped_rgbds], dim=1)
        reconst_img, feats = model.feature_net(input_imgs.reshape(args.num_source_views, -1, H, W))
        featmaps = {}
        featmaps['coarse'] = feats[:, :args.coarse_feat_dim]
        featmaps['fine']   = feats[:, args.coarse_feat_dim:]
        src_rgbs = ray_sampler.src_rgbs.to(device)
        input_src_rgbs_ = reconst_img.permute(0,2,3,1)[None]

        all_ret['depth_warped_imgs'] = input_imgs[:2, :, :3]

    else:
        pred_offset = None
        pred_kernel = None
        nearby_idxs = None
        reconst_img = None
        depth = None

        input_src_rgbs_ = input_src_rgbs
        input_src_rgbs  = input_src_rgbs
        src_rgbs, featmaps = ray_render.calc_featmaps(input_src_rgbs)

    if reconst_img != None:
        all_ret['kernel_reconst'] = reconst_img

    if depth != None:
        all_ret['patchmatch_depth'] = depth
        

    if args.N_importance > 0:
        all_ret['fine'] = RaysOutput.empty_ret()
    N_rays = ray_sampler.rays_o.shape[0]

    H, W = input_src_rgbs.shape[-3:-1]
    for i in tqdm(range(0, N_rays, args.chunk_size)):
        # print('batch', i)
        ray_batch = ray_sampler.specific_ray_batch(slice(i, i + args.chunk_size, 1), clean=args.sup_clean)
        if False: #'pred_offset' in featmaps.keys():
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
                                            org_src_rgbs=input_src_rgbs_,
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




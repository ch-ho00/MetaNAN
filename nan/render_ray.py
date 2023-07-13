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
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

from nan.model import NANScheme
from nan.projection import Projector
from nan.raw2output import RaysOutput
from nan.feature_network import BasicBlock
from nan.sample_ray import parse_camera
from nan.dataloaders.basic_dataset import de_linearize


def stack_image(image, N=2, pad=7):
    b, c, h, w = image.size()
    assert h % N == 0 and w % 8 == 0
    padded_image = torch.nn.functional.pad(image, (pad, pad, pad, pad), mode='constant', value=0)
    patch_h = h//N
    patch_w = w//N 
    stacked = torch.zeros((b * N * N, c, patch_h + 2*pad, patch_w + 2*pad)).to(image.device)
    for j in range(N):
        for k in range(N):
            row_start = j* patch_h
            row_end = (j+1)* patch_h + 2*pad
            col_start = k* patch_w
            col_end = (k+1)* patch_w + 2*pad
            stacked[(j * N + k) * b: (j * N + k + 1) * b] = padded_image[:, :, row_start:row_end, col_start:col_end]
    return stacked


def unstack_image(stack_image, total_n_patch=4, pad=7):
    b, c, patch_h, patch_w = stack_image.size()
    assert b % total_n_patch == 0
    N = int(total_n_patch ** 0.5)
    
    out_patch_h = patch_h - pad * 2
    out_patch_w = patch_w - pad * 2
    nimgs = b // total_n_patch
    output = torch.zeros((nimgs, c, out_patch_h * N, out_patch_w * N)).to(stack_image.device)
    # print(output.shape, stack_image.shape)
    for patch_idx in range(b):
        h_idx = (patch_idx // nimgs) // N
        w_idx = (patch_idx // nimgs) % N
        
        start_h = pad
        end_h   = pad + out_patch_h

        start_w = pad
        end_w   = pad + out_patch_w
        
        # print(patch_idx, patch_idx % nimgs, h_idx, w_idx)
        # print(start_h, end_h, 0, patch_h)
        # print(start_w, end_w, 0, patch_w)
        # print(output[patch_idx // total_n_patch, :, out_patch_h * h_idx: out_patch_h * (h_idx + 1), out_patch_w * w_idx : out_patch_w * (w_idx +1)].shape, 
        #       stack_image[patch_idx, :, start_h: end_h, start_w: end_w].shape)
        output[patch_idx % nimgs, :, out_patch_h * h_idx: out_patch_h * (h_idx + 1), out_patch_w * w_idx : out_patch_w * (w_idx +1)] = stack_image[patch_idx, :, start_h: end_h, start_w: end_w]
    return output



def sample_pdf(bins, weights, N_samples, det=False):
    """
    @param: bins: tensor of shape [N_rays, M+1], M is the number of bins
    @param: weights: tensor of shape [N_rays, M]
    @param: N_samples: number of samples along each ray
    @param: det: if True, will perform deterministic sampling
    @return: [N_rays, N_samples]
    """

    N_rays, M = weights.shape
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1)  # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).expand(bins.shape[0], -1)  # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)  # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i + 1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)  # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).expand(N_rays, N_samples, M + 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).expand(N_rays, N_samples, M + 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]  # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1] - bins_g[:, :, 0])

    return samples


class RayRender:
    """
    Object that handle the actual rendering:
    * Calculating features maps
    * Sampling 3D points along a ray
    * Projecting them to each of the other views
    * Feeding the network with the incident pixels
    * Rendering rho and RGB values to each 3D point along a ray
    * Aggregate the values from 3D points along a specific ray
    """
    def __init__(self, model: NANScheme, args, device, save_pixel=None):
        self.model = model
        self.device = device
        self.projector = Projector(device=device, args=args)

        self.N_samples = args.N_samples
        self.inv_uniform = args.inv_uniform
        self.N_importance = args.N_importance
        self.det = args.det
        self.white_bkgd = args.white_bkgd

        # For debug purposes
        if save_pixel is not None:
            y, x = tuple(zip(*save_pixel))
            self.save_pixels = torch.tensor((x, y, (1,) * len(x)))
        else:
            self.save_pixels = None

        self.fine_processing = args.N_importance > 0
        if self.fine_processing:
            assert self.model.net_fine is not None

    def pixel2index(self, ray_batch):
        if self.save_pixels is not None:
            row_detection = (ray_batch['xyz'].unsqueeze(-1) == self.save_pixels.unsqueeze(0)).all(1)
            idx_in_batch, idx_in_all = torch.where(row_detection)
            if len(idx_in_batch) > 0:
                return tuple(zip(*(idx_in_batch.tolist(), self.save_pixels[:2:, idx_in_all].T.flip(1).tolist())))

    def sample_along_blur_ray_coarse(self, ray_batch, featmaps):
        """
        :param: ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
        :param: ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
        :param: depth_range: [near_depth, far_depth]
        :param: inv_uniform: if True, uniformly sampling inverse depth
        :param: det: if True, will perform deterministic sampling
        :return: 3D point: tensor of shape [N_rays, N_samples, 3], z_vals: tensor of shape [N_rays, N_samples, 3]
        """
        # will sample inside [near_depth, far_depth]
        # assume the nearest possible depth is at least (min_ratio * depth)
        device = featmaps['tar_bpn_feats'].device
        h8, w8 = featmaps['tar_bpn_feats'].shape[-2:]
        h, w = h8 * 8, w8 * 8
        pixel_coords = ray_batch['xyz'][:,:2].float() 
        pixel_coords[:,0] = 2 * (pixel_coords[:,0] / w - 0.5)
        pixel_coords[:,1] = 2 * (pixel_coords[:,1] / h - 0.5)
        pixel_coords = pixel_coords.to(device)
        img_feats = self.model.img_embed_conv(featmaps['tar_bpn_feats'])
        img_embed = F.grid_sample(img_feats, pixel_coords[None, None], mode='bilinear')
        img_embed = img_embed.squeeze().permute(1,0)
        img_embed = torch.cat([img_embed, featmaps['tar_noise_vec'].repeat(img_embed.shape[0], 1)], dim=-1)

        offset_output = self.model.blur_kernel_fc(img_embed)
        delta_trans, delta_pos, weight = torch.split(offset_output, [10, 10, 5], dim=-1)
        delta_trans  = delta_trans.reshape(-1, self.model.num_kernel_pt, 2)
        delta_pos    = delta_pos.reshape(-1, self.model.num_kernel_pt, 2)  
        weight       = weight.reshape(-1, self.model.num_kernel_pt, 1)     
        weight = torch.softmax(weight[..., 0], dim=-1)

        delta_trans = delta_trans * 0.01

        rays_x, rays_y = torch.split(ray_batch['xyz'][:,:2].float().to(device), [1,1], dim=-1)
        W, H, K, poses = parse_camera(ray_batch['camera'].to(device))
        K = K[0]
        rays_x = (rays_x - K[0, 2] + delta_pos[..., 0]) / K[0, 0]
        rays_y = (rays_y - K[1, 2] + delta_pos[..., 1]) / K[1, 1]
        dirs = torch.stack([rays_x - delta_trans[..., 0],
                            rays_y - delta_trans[..., 1],
                            torch.ones_like(rays_x)], -1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * poses[..., None, :3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        translation = torch.stack([
            delta_trans[..., 0],
            delta_trans[..., 1],
            torch.zeros_like(rays_x),
            torch.ones_like(rays_x)
        ], dim=-1)
        rays_o = torch.sum(translation[..., None, :] * poses[:, None, :3], dim=-1)

        align = delta_pos[:, 0, :].abs().mean()
        align += (delta_trans[:, 0, :].abs().mean() * 10)

        depth_range = ray_batch['depth_range'].to(device)
        near_depth_value = depth_range[0, 0]
        far_depth_value = depth_range[0, 1]
        assert 0 < near_depth_value < far_depth_value and far_depth_value > 0

        near_depth = near_depth_value * torch.ones_like(rays_d[..., 0, 0])
        far_depth = far_depth_value * torch.ones_like(rays_d[..., 0, 0])

        if self.inv_uniform:
            start = 1. / near_depth  # [N_rays,]
            step = (1. / far_depth - start) / (self.N_samples - 1)
            inv_z_vals = torch.stack([start + i * step for i in range(self.N_samples)], dim=1)  # [N_rays, N_samples]
            z_vals = 1. / inv_z_vals
        else:
            start = near_depth
            step = (far_depth - near_depth) / (self.N_samples - 1)
            z_vals = torch.stack([start + i * step for i in range(self.N_samples)], dim=1)  # [N_rays, N_samples]

        if not self.det:
            # get intervals between samples
            mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
            upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
            lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
            # uniform samples in those intervals
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

        rays_o = rays_o.reshape(-1,3)
        rays_d = rays_d.reshape(-1,3)
        z_vals = z_vals[:,None].repeat(1,self.model.num_kernel_pt,1).reshape(-1, z_vals.shape[-1])        
        pts = z_vals[...,None] * rays_d.unsqueeze(1) + rays_o.unsqueeze(1)  # [N_rays, N_samples, 3]

        return pts, z_vals, rays_o, rays_d, weight, align

    def sample_along_ray_coarse(self, ray_o, ray_d, depth_range):
        """
        :param: ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
        :param: ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
        :param: depth_range: [near_depth, far_depth]
        :param: inv_uniform: if True, uniformly sampling inverse depth
        :param: det: if True, will perform deterministic sampling
        :return: 3D point: tensor of shape [N_rays, N_samples, 3], z_vals: tensor of shape [N_rays, N_samples, 3]
        """
        # will sample inside [near_depth, far_depth]
        # assume the nearest possible depth is at least (min_ratio * depth)
        near_depth_value = depth_range[0, 0]
        far_depth_value = depth_range[0, 1]
        assert 0 < near_depth_value < far_depth_value and far_depth_value > 0

        near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])
        far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])

        if self.inv_uniform:
            start = 1. / near_depth  # [N_rays,]
            step = (1. / far_depth - start) / (self.N_samples - 1)
            inv_z_vals = torch.stack([start + i * step for i in range(self.N_samples)], dim=1)  # [N_rays, N_samples]
            z_vals = 1. / inv_z_vals
        else:
            start = near_depth
            step = (far_depth - near_depth) / (self.N_samples - 1)
            z_vals = torch.stack([start + i * step for i in range(self.N_samples)], dim=1)  # [N_rays, N_samples]

        if not self.det:
            # get intervals between samples
            mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
            upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
            lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
            # uniform samples in those intervals
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

        pts = z_vals.unsqueeze(2) * ray_d.unsqueeze(1) + ray_o.unsqueeze(1)  # [N_rays, N_samples, 3]
        return pts, z_vals

    def sample_along_ray_fine(self, coarse_out: RaysOutput, z_vals, ray_batch):
        # detach since we would like to decouple the coarse and fine networks
        weights = coarse_out.weights.clone().detach()  # [N_rays, N_samples]

        if self.inv_uniform:
            inv_z_vals = 1. / z_vals
            inv_z_vals_mid = .5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])  # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
            inv_z_vals = sample_pdf(bins=torch.flip(inv_z_vals_mid, dims=[1]),
                                    weights=torch.flip(weights, dims=[1]),
                                    N_samples=self.N_importance, det=self.det)  # [N_rays, N_importance]
            z_samples = 1. / inv_z_vals
        else:
            # take mid-points of depth samples
            z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])  # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]  # [N_rays, N_samples-2]
            z_samples = sample_pdf(bins=z_vals_mid, weights=weights,
                                   N_samples=self.N_importance, det=self.det)  # [N_rays, N_importance]
        z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]

        # samples are sorted with increasing depth
        z_vals, _ = torch.sort(z_vals, dim=-1)
        N_total_samples = self.N_samples + self.N_importance
        N_rays = weights.shape[0]
        viewdirs = ray_batch['ray_d'].unsqueeze(1).expand(N_rays, N_total_samples, 3)
        ray_o = ray_batch['ray_o'].unsqueeze(1).expand(N_rays, N_total_samples, 3)
        pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]
        return pts, z_vals

    def render_batch(self, ray_batch, proc_src_rgbs, featmaps, org_src_rgbs,
                     sigma_estimate, blur_target=False) -> Dict[str, RaysOutput]:
        """
        :param sigma_estimate: (1, N, H, W, 3)
        :param org_src_rgbs: (1, N, H, W, 3)
        :param proc_src_rgbs: org_src_rgbs after processing with pre_net, if exists (1, N, H, W, 3)
        :param featmaps: dict {'corase': (N, C, H', W'), 'fine': (N, C, H', W')}
        :param ray_batch: {ray_o: (R, 3),
                           ray_d: (R, 3),
                           camera: (1, 34),
                           depth_range: (1, 2),
                           src_cameras: (1, N, 34),
                           selected_inds: (R,),
                           xyz: (R, 3),
                           rgb: (R, 3),
                           white_level: (1, 1)}
        :return: {'coarse': {}, 'fine': {}}
        """

        # Find pixels for debug
        save_idx = self.pixel2index(ray_batch)

        # Create the output dictionary
        batch_out = {'coarse': None,
                     'fine': None}

        # pts:    [R, S, 3]
        # z_vals: [R, S]
        # Sample points along ray for coarse phase
        blur_ray_weights = None 
        if self.model.args.blur_render and blur_target:
            pts_coarse, z_vals_coarse, rays_o, rays_d, blur_ray_weights, align = self.sample_along_blur_ray_coarse(ray_batch=ray_batch, featmaps=featmaps)            
            ray_batch['ray_o'] = rays_o
            ray_batch['ray_d'] = rays_d
        else:
            pts_coarse, z_vals_coarse = self.sample_along_ray_coarse(ray_o=ray_batch['ray_o'],
                                                                    ray_d=ray_batch['ray_d'],
                                                                    depth_range=ray_batch['depth_range'])

        # Process the rays and return the coarse phase output
        coarse_ray_out = self.process_rays_batch(ray_batch=ray_batch, pts=pts_coarse, z_vals=z_vals_coarse, save_idx=save_idx,
                                         level='coarse', proc_src_rgbs=proc_src_rgbs, featmaps=featmaps,
                                         org_src_rgbs=org_src_rgbs, sigma_estimate=sigma_estimate, blur_target=blur_target, blur_ray_weights=blur_ray_weights)
        batch_out['coarse'] = coarse_ray_out

        if self.fine_processing:
            # Sample points along ray for fine phase, based on the coarse output
            pts_fine, z_vals_fine = self.sample_along_ray_fine(coarse_out=coarse_ray_out,
                                                               z_vals=z_vals_coarse,
                                                               ray_batch=ray_batch)

            # Process the rays and return the fine phase output
            fine = self.process_rays_batch(ray_batch=ray_batch, pts=pts_fine, z_vals=z_vals_fine, save_idx=save_idx,
                                           level='fine', proc_src_rgbs=proc_src_rgbs, featmaps=featmaps,
                                           org_src_rgbs=org_src_rgbs, sigma_estimate=sigma_estimate, blur_target=blur_target, blur_ray_weights=blur_ray_weights)

            batch_out['fine'] = fine

            if self.model.args.blur_render and blur_target:
                n_samples_coarse = batch_out['coarse'].weights.shape[-1]
                n_samples_fine = batch_out['fine'].weights.shape[-1]
                batch_out['coarse'].rgb = torch.sum(batch_out['coarse'].rgb.reshape(-1, self.model.num_kernel_pt, 3) * blur_ray_weights[...,None], dim=1)
                batch_out['coarse'].weights = torch.sum(batch_out['coarse'].weights.reshape(-1, self.model.num_kernel_pt, n_samples_coarse) * blur_ray_weights[...,None], dim=1)
                batch_out['coarse'].depth = torch.sum(batch_out['coarse'].depth.reshape(-1, self.model.num_kernel_pt) * blur_ray_weights, dim=1)
                batch_out['coarse'].mask = torch.sum(batch_out['coarse'].mask.reshape(-1, self.model.num_kernel_pt), dim=1)

                batch_out['fine'].rgb = torch.sum(batch_out['fine'].rgb.reshape(-1, self.model.num_kernel_pt, 3) * blur_ray_weights[...,None], dim=1)
                batch_out['fine'].weights = torch.sum(batch_out['fine'].weights.reshape(-1, self.model.num_kernel_pt, n_samples_fine) * blur_ray_weights[...,None], dim=1)
                batch_out['fine'].depth = torch.sum(batch_out['fine'].depth.reshape(-1, self.model.num_kernel_pt) * blur_ray_weights, dim=1)
                batch_out['fine'].mask = torch.sum(batch_out['fine'].mask.reshape(-1, self.model.num_kernel_pt), dim=1)

                batch_out['align_loss'] = align


        return batch_out

    def process_rays_batch(self, ray_batch, pts, z_vals, save_idx, level, proc_src_rgbs, featmaps,
                           org_src_rgbs, sigma_estimate, blur_target=False, blur_ray_weights=None):
        """
        :param sigma_estimate: (1, N, H, W, 3)
        :param org_src_rgbs: (1, N, H, W, 3)
        :param proc_src_rgbs: (1, N, H, W, 3)
        :param featmaps: (N, 3, H', W')
        :param level: str {'coarse', 'fine'}
        :param ray_batch: dictionary with rays origin, direction and relevant data for rendering
        :param pts: 3D points along the rays (R, S, 3)
        :param z_vals: z values from which the 3D points were calcuated (R, S)
        :param save_idx: indices for debug purposes
        :return: RaysOutput object of the rendered values
        """
        # Project the pts along the rays batch on all others views (src views)
        # based on the target camera and src cameras (intrinsics - K, rotation - R, translation - t)
        proj_out = self.projector.compute(pts, ray_batch['camera'], proc_src_rgbs, org_src_rgbs, sigma_estimate,
                                          ray_batch['src_cameras'],
                                          featmaps=featmaps[level])  # [N_rays, N_samples, N_views, x]
        rgb_feat, ray_diff, pts_mask, org_rgb, sigma_est, proj_feat = proj_out

        # [N_rays, N_samples, 4]
        # Process the feature vectors of all 3D points along each ray to predict density and rgb value
        rgb_out, rho_out, *debug_info = self.model.mlps[level](rgb_feat, ray_diff,
                                                               pts_mask.unsqueeze(-3).unsqueeze(-3),
                                                               org_rgb, sigma_est, featmaps['noise_vec'])
        ray_outputs = RaysOutput.raw2output(rgb_out, rho_out, z_vals, pts_mask, white_bkgd=self.white_bkgd)

        if save_idx is not None:
            debug_dict = {}
            for idx, pixel in save_idx:
                debug_dict[(tuple(pixel))] = OrderedDict([('z', ray_outputs.z_vals[idx].cpu()),
                                                          ('w', ray_outputs.weights[idx].cpu()),
                                                          ('w_rgb', debug_info[0][idx].cpu() if debug_info[0][idx] != None else None),
                                                          ('feat', debug_info[1][idx].cpu()),
                                                          ('globalfeat_attention', debug_info[2][idx].cpu())])
            ray_outputs.debug = debug_dict

        return ray_outputs

    def calc_featmaps(self, src_rgbs, sigma_estimate=None, white_level=None):
        """
        Calculating the features maps of the source views
        :param src_rgbs: (1, N, H, W, 3)
        :return: src_rgbs after pre_net (if exists) (1, N, H, W, 3),
                 features maps: dict {'coarse': (N, C, H', W'), 'fine': (N, C, H', W')}
        """
        conv1_weights = None
        orig_rgbs = src_rgbs
        featmaps = {}
        if self.model.pre_net is not None:
            if self.model.args.bpn_prenet:
                pad = 8
                npatch_per_side = 2
                src_rgbs = src_rgbs.squeeze(0).permute(0, 3, 1, 2)
                src_rgbs_stacked = stack_image(src_rgbs, N=npatch_per_side, pad=pad)
                src_rgbs, bpn_feats = self.model.pre_net(src_rgbs_stacked, src_rgbs_stacked[:,None])                    
                src_rgbs = src_rgbs[:,0]
                del src_rgbs_stacked
                src_rgbs = unstack_image(src_rgbs, total_n_patch=npatch_per_side**2, pad=pad)
                bpn_feats = unstack_image(bpn_feats, total_n_patch=npatch_per_side**2, pad=1)
                featmaps['bpn_feats'] = bpn_feats
                torch.cuda.empty_cache()
            else:
                src_rgbs = self.model.pre_net(src_rgbs.squeeze(0).permute(0, 3, 1, 2))  # (N, 3, H, W)


        noise_vec = None
        if self.model.args.cond_renderer:
            H, W = orig_rgbs.shape[2:4]
            start_h = 0 if H < 384 else (H - 384) // 2
            start_w = 0 if W < 384 else (W - 384) // 2
            input_rgb = orig_rgbs[0, :, start_h:start_h + 384, start_w: start_w + 384].permute(0,3,1,2)
            white_level = white_level.to(orig_rgbs.device)
            if white_level.ndim == 2 and white_level.shape[0] == 1:
                white_level = white_level[0].item()
            x = de_linearize(input_rgb, white_level)
            x = self.model.degae.degrep_extractor.srgan(x)
            noise_vec = self.model.degae.degrep_extractor.degrep_conv(x)
            noise_vec = F.adaptive_avg_pool2d(noise_vec, (1, 1))
            noise_vec = self.model.degae.degrep_extractor.degrep_fc(noise_vec.reshape(-1,512))
            del input_rgb, x
            torch.cuda.empty_cache()

        if not self.model.args.degae_feat:
            featmaps = self.model.feature_net(src_rgbs)
        else:
            with torch.no_grad():
                degfeat = self.model.degae.encoder(orig_rgbs[0].permute(0,3,1,2), img_wh=torch.Tensor([orig_rgbs.shape[-2], orig_rgbs.shape[-3]]).int().to(orig_rgbs.device))    
            degfeat = self.model.feature_conv_0(degfeat) 
            degfeat = self.model.feature_conv_1(degfeat) 
            degfeat = self.model.feature_conv_2(degfeat)
            feat    = self.model.feature_conv_3(degfeat)

            featmaps['coarse']  = feat[:,:self.model.args.coarse_feat_dim]
            featmaps['fine']    = feat[:,self.model.args.coarse_feat_dim:]
            del degfeat

        src_rgbs = src_rgbs.permute(0, 2, 3, 1).unsqueeze(0)
        featmaps['noise_vec'] = noise_vec

        return src_rgbs , featmaps

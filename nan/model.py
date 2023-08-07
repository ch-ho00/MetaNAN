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

# #### Modified version
# #### see https://github.com/googleinterns/IBRNet for original

from pathlib import Path
from typing import Tuple, Dict

import kornia
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from configs.local_setting import OUT_DIR
from nan.feature_network import ResUNet
from nan.nan_mlp import NanMLP
from nan.utils.io_utils import get_latest_file, print_link
from degae.model import DegAE
from degae.decoder import BasicBlock
from nan.bpn_prenet import BPN, DeblurBPN

def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


def parallel(model, local_rank):
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

class Gaussian2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], sigma: Tuple[float, float]):
        super().__init__(in_channels, out_channels, kernel_size, padding='same', bias=True)

        gauss_kernel: torch.Tensor = kornia.filters.get_gaussian_kernel2d(kernel_size, sigma)
        new_weight = torch.zeros_like(self.weight)
        new_weight[0, 0] = gauss_kernel
        new_weight[1, 1] = gauss_kernel
        new_weight[2, 2] = gauss_kernel

        with torch.no_grad():
            self.weight.copy_(new_weight)
        # nn.init.zeros_(self.bias.data)



def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))

        self.downsample = downsample
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        out_down = self.downsample(out)
        
        return out_down


class NoiseLevelConv(nn.Module):
    def __init__(self):
        super(NoiseLevelConv, self).__init__()

        self.out_dim = 256
        self.out_size = 8
        self.conv0 = UNetConvBlock(3, 64, True, 0.2)
        self.conv1 = UNetConvBlock(64, 128, True, 0.2)
        self.conv2 = UNetConvBlock(128, 256, True, 0.2)
        self.conv3 = UNetConvBlock(256, self.out_dim, True, 0.2)

    def forward(self, x):
        x = self.conv0(x) # (B, 32, H//2, W//2)
        x = self.conv1(x) # (B, 64, H//4, W//4)
        x = self.conv2(x) # (B, 128, H//8, W//8)
        x = self.conv3(x) # (B, 256, H//16, W//16)
        x = F.adaptive_avg_pool2d(x, (self.out_size, self.out_size))
        return x


class ConvWeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, patch_kernel=False):
      super(ConvWeightGenerator,self).__init__()
      self.in_dim = in_dim
      self.out_dim = out_dim
      self.patch_kernel = patch_kernel

      self.transform = nn.Sequential(
        nn.Linear(self.in_dim, 4096),
        nn.LeakyReLU(0.2),
        nn.Linear(4096,4096),
        nn.LeakyReLU(0.2),
        nn.Linear(4096,self.out_dim)
      )

    def forward(self,noise_vec):
      if self.patch_kernel:
        noise_vec = noise_vec.reshape(noise_vec.shape[0], self.in_dim, -1).permute(0,2,1)
      weights = self.transform(noise_vec)
      return weights 


class NANScheme(nn.Module):
    @classmethod
    def create(cls, args):
        model = cls(args)
        if args.distributed:
            model = parallel(model, args.local_rank)
        return model

    def __init__(self, args):
        super().__init__()
        self.args = args
        device = torch.device(f'cuda:{args.local_rank}')

        # create feature extraction network
        self.degae_feat = args.degae_feat
        if args.degae_feat or args.lambda_embed_loss > 0 or args.cond_renderer:
            self.degae = DegAE(args, train_scratch=False)
            if args.degae_feat_ckpt != None:
                checkpoint = torch.load(args.degae_feat_ckpt, map_location=lambda storage, loc: storage)
                self.degae.load_state_dict(checkpoint["model"])
            
            for param in self.degae.parameters():
                param.requires_grad = False
            self.degae.eval()

            if args.degae_feat:
                dim_ = self.args.fine_feat_dim + self.args.coarse_feat_dim
                self.feature_conv_0 = BasicBlock(dim_, dim_, stride=2, downsample=True,  rand_noise=True).to(device)
                self.feature_conv_1 = BasicBlock(dim_, dim_, stride=1, downsample=None, rand_noise=True).to(device)
                self.feature_conv_2 = BasicBlock(dim_, dim_, stride=2, downsample=True,  rand_noise=True).to(device)
                self.feature_conv_3 = BasicBlock(dim_, dim_, stride=1, downsample=None, rand_noise=True).to(device)


        if not args.degae_feat:
            self.feature_net = ResUNet(coarse_out_ch=args.coarse_feat_dim,
                                    fine_out_ch=args.fine_feat_dim,
                                    coarse_only=args.coarse_only).to(device)


        # create coarse NAN mlps
        self.net_coarse = self.nan_factory('coarse', device)
        self.net_fine = None

        if not args.coarse_only:
            # create fine NAN mlps
            self.net_fine = self.nan_factory('fine', device)

        self.mlps: Dict[str, NanMLP] = {'coarse': self.net_coarse, 'fine': self.net_fine}

        if args.pre_net:
            if args.bpn_prenet:
                if args.blur_render:
                    self.pre_net = DeblurBPN(n_latent_layers=args.num_latent, burst_length=1, group_conv=args.group_conv, channel_upfactor=args.channel_upfactor, skip_connect=args.skip_connect).to(device)
                else:
                    self.pre_net = BPN(bpn_per_img=args.bpn_per_img, burst_length=1).to(device)
            else:
                if args.weightsum_filtered:
                    self.pre_net = Gaussian2D(in_channels=3, out_channels=3, kernel_size=(13, 13), sigma=(1.5, 1.5)).to(device)                
                else:
                    self.pre_net = Gaussian2D(in_channels=3, out_channels=3, kernel_size=(3, 3), sigma=(1.5, 1.5)).to(device)
        else:
            self.pre_net = None
            

        out_folder = OUT_DIR / args.expname

        # optimizer and learning rate scheduler
        self.optimizer, self.scheduler = self.create_optimizer()

        self.start_step = self.load_from_ckpt(out_folder)

    def create_optimizer(self):
        params_list = []
        if self.args.degae_feat or self.args.cond_renderer:
            if self.args.degae_feat:
                params_list += [ {'params': self.feature_conv_0.parameters(), 'lr': self.args.lrate_feature},
                                {'params': self.feature_conv_1.parameters(), 'lr': self.args.lrate_feature},
                                {'params': self.feature_conv_2.parameters(), 'lr': self.args.lrate_feature},
                                {'params': self.feature_conv_3.parameters(), 'lr': self.args.lrate_feature}]     
            if self.args.ft_embed_fc:
                params_list += [{'params' : self.degae.degrep_extractor.degrep_conv.parameters(), 'lr':self.args.lrate_feature * 1e-2},
                                {'params' : self.degae.degrep_extractor.degrep_fc.parameters(),   'lr':self.args.lrate_feature * 1e-2}]
                    
        else:
            params_list = [{'params': self.feature_net.parameters(), 'lr': self.args.lrate_feature}]

        params_list.append( {'params': self.net_coarse.parameters(),  'lr': self.args.lrate_mlp})

        if self.net_fine is not None:
            params_list.append({'params': self.net_fine.parameters(), 'lr': self.args.lrate_mlp})

        if self.args.blur_render and self.args.bpn_prenet:
            params_list.append({'params': self.pre_net.bpn.parameters(), 'lr': self.args.lrate_feature})
            params_list.append({'params': self.pre_net.offset_conv.parameters(), 'lr': self.args.lrate_feature * 1e-1})                
            # params_list.append({'params': self.pre_net.offset_fc.parameters(), 'lr': self.args.lrate_feature * 1e-2})                
        elif self.args.pre_net:
            params_list.append({'params': self.pre_net.parameters(), 'lr': self.args.lrate_feature})



        optimizer = torch.optim.Adam(params_list)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.args.lrate_decay_steps,
                                                    gamma=self.args.lrate_decay_factor)

        return optimizer, scheduler

    def switch_to_eval(self):
        self.net_coarse.eval()
        if self.args.degae_feat:
            self.feature_conv_0.eval()        
            self.feature_conv_1.eval()        
            self.feature_conv_2.eval()        
            self.feature_conv_3.eval()        


        else:
            self.feature_net.eval()

        if self.net_fine is not None:
            self.net_fine.eval()

        if self.pre_net is not None:
            self.pre_net.eval()
            if self.args.blur_render and self.args.bpn_prenet:
                self.pre_net.offset_conv.eval()
                # self.pre_net.offset_fc.eval()

        if self.args.ft_embed_fc:
            self.degae.degrep_extractor.degrep_conv.eval()
            self.degae.degrep_extractor.degrep_fc.eval()


    def switch_to_train(self):
        self.net_coarse.train()
        if self.args.degae_feat:
            self.feature_conv_0.train()        
            self.feature_conv_1.train()        
            self.feature_conv_2.train()        
            self.feature_conv_3.train()        

        else:
            self.feature_net.train()

        if self.net_fine is not None:
            self.net_fine.train()

        if self.pre_net is not None:
            self.pre_net.train()
            if self.args.blur_render and self.args.bpn_prenet:
                self.pre_net.offset_conv.train()
                # self.pre_net.offset_fc.train()

        if self.args.ft_embed_fc:
            self.degae.degrep_extractor.degrep_conv.train()
            self.degae.degrep_extractor.degrep_fc.train()



    def save_model(self, filename):
        to_save = {'optimizer'  : self.optimizer.state_dict(),
                   'scheduler'  : self.scheduler.state_dict(),
                   'model' : de_parallel(self).state_dict()}
        torch.save(to_save, filename)

    def load_model(self, filename):
        load_dict = torch.load(filename, map_location=torch.device(f'cuda:{self.args.local_rank}'))
        if 'model' not in load_dict:
            # for old version of ckpt
            load_dict = self.convert_state_to_model(load_dict)

        model_dict = load_dict['model'].copy()
        for key in load_dict['model']:
            if "spatial_views_attention" in key:
                print(f"[**] removing key {key}")
                del model_dict[key]

        load_dict['model'] = model_dict

        if not self.args.no_load_opt:
            self.optimizer.load_state_dict(load_dict['optimizer'])
        if not self.args.no_load_scheduler:
            self.scheduler.load_state_dict(load_dict['scheduler'])

        self.load_weights_to_net(self, load_dict['model'], self.args.allow_weights_mismatch)

    @staticmethod
    def convert_state_to_model(load_dict):
        new_load_dict = {'optimizer': load_dict['optimizer'], 'scheduler': load_dict['scheduler'], 'model': {}}
        for net, weights in load_dict.items():
            if net not in ['optimizer', 'scheduler']:
                new_load_dict['model'].update({f"{net}.{k}": w for k, w in weights.items()})

        return new_load_dict

    @staticmethod
    def load_weights_to_net(net, pretrained_dict, allow_weights_mismatch):
        try:
            net.load_state_dict(pretrained_dict)
        except RuntimeError:
            if not allow_weights_mismatch:
                raise
            else:
                # from https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
                new_model_dict = net.state_dict()

                # 1. filter of weights with shape mismatch
                for pre_k, pre_v in pretrained_dict.items():
                    if pre_k in new_model_dict:
                        new_v = new_model_dict[pre_k]
                        if new_v.shape == pre_v.shape:
                            new_model_dict[pre_k] = new_v
                        else:
                            pass
                            # if we want to load partial layers, it can be done with:
                            # new_model_dict[pre_k][torch.where(torch.ones_like(new_v))] = new_v.view(-1).clone()
                # 3. load the new state dict
                net.load_state_dict(new_model_dict)

    def load_from_ckpt(self, out_folder: Path):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpt = None

        if out_folder.exists() and self.args.resume_training:
            print_link(out_folder, "[*] Resume training looking for ckpt in ")
            try:
                ckpt = get_latest_file(out_folder, "*.pth")
            except ValueError:
                pass

        if self.args.ckpt_path is not None and not self.args.resume_training:
            if not self.args.ckpt_path.exists():  # load the specified ckpt
                raise FileNotFoundError(f"requested ckpt_path does not exist: {self.args.ckpt_path}")
            ckpt = self.args.ckpt_path

        if ckpt is not None and not self.args.no_reload:
            step = int(ckpt.stem[-6:])
            print_link(ckpt, '[*] Reloading from', f"starting at step={step}")
            self.load_model(ckpt)
        else:
            if ckpt is None:
                print('[*] No ckpts found, training from scratch...')
                print(str(self.args.ckpt_path))
            if self.args.no_reload:
                print('[*] no_reload, training from scratch...')

            step = 0

        return step

    def nan_factory(self, net_type, device) -> NanMLP:
        if net_type == 'coarse':
            feat_dim = self.args.coarse_feat_dim
            n_samples = self.args.N_samples
        elif net_type == 'fine':
            feat_dim = self.args.fine_feat_dim
            n_samples = self.args.N_samples + self.args.N_importance
        else:
            raise NotImplementedError

        return NanMLP(self.args,
                      in_feat_ch=feat_dim,
                      n_samples=n_samples).to(device)


if __name__ == '__main__':
    features_net = nn.Linear(5, 10)
    mlp = nn.Linear(5, 10)

    # optimizer and learning rate scheduler
    p_list = [{'params': features_net.parameters(), 'lr': 0.001},
              {'params': mlp.parameters(), 'lr': 0.0005}]

    optimizer_ = torch.optim.Adam(p_list)

    # scheduler_ = torch.optim.lr_scheduler.LambdaLR(optimizer_, lr_lambda=[lambda step: 0.999 ** step,
    #                                                                       lambda step: 0.9999 ** (4000 - step)])
    lrate_decay_steps = 10000
    lrate_decay_factor = 0.5

    def lr_schedule(step):
        if step < 10000:
            return 1
        else:
            return lrate_decay_factor ** ((step - 10000) // lrate_decay_steps)

    scheduler_ = torch.optim.lr_scheduler.LambdaLR(optimizer_, lr_lambda=lr_schedule)

    lr_list = []
    for i in range(40000):
        optimizer_.zero_grad()
        optimizer_.step()

        lr_list.append(scheduler_.get_last_lr())
        if i % 5000 == 0:
            print(i, scheduler_.get_last_lr())
        scheduler_.step()

    lr_feat, lr_mlp = zip(*lr_list)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('steps')
    ax1.set_ylabel('lr features', color=color)
    ax1.plot(lr_feat, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('lr mlp', color=color)  # we already handled the x-label with ax1
    ax2.plot(lr_mlp, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()



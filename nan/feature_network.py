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
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='reflect')


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='reflect')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class Upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super().__init__()
        self.scale = scale
        self.conv = Conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self,
                 encoder='resnet34',
                 coarse_out_ch=32,
                 fine_out_ch=32,
                 norm_layer=None,
                 coarse_only=False,
                 auto_encoder=False,
                 per_level_render=False,
                 meta_module=False):

        super().__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        self.coarse_only = coarse_only
        if self.coarse_only:
            fine_out_ch = 0
        self.coarse_out_ch = coarse_out_ch
        self.fine_out_ch = fine_out_ch
        out_ch = coarse_out_ch + fine_out_ch

        self.meta_module = meta_module
        # original
        layers = [3, 4, 6, 3]
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        block = BasicBlock
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        if not self.meta_module:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False, padding_mode='reflect')
        else:
            self.conv1_in_dim = self.inplanes // 2
            self.conv1_kdim = self.conv1_in_dim * 3 * 7 * 7 
            self.conv1_half = nn.Conv2d(3, self.conv1_in_dim, kernel_size=7, stride=2, padding=3,
                                bias=False, padding_mode='reflect')

        self.bn1 = norm_layer(self.inplanes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # decoder
        self.upconv3 = Upconv(filters[2], 128, 3, 2)
        self.iconv3 = Conv(filters[1] + 128, 128, 3, 1)
        self.upconv2 = Upconv(128, 64, 3, 2)
        self.iconv2 = Conv(filters[0] + 64, out_ch, 3, 1)

        # fine-level conv
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1, 1)

        self.auto_encoder = auto_encoder
        self.per_level_render = per_level_render
        if self.auto_encoder:
            deconv_in_dim = out_ch // 2
            self.reconst_deconv =  nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(deconv_in_dim, out_ch//2, 3, 1, 1),
                nn.BatchNorm2d(out_ch//2),
                nn.LeakyReLU(0.2, True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(out_ch//2, out_ch//4, 3, 1, 1),
                nn.BatchNorm2d(out_ch//4),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_ch//4, 3, 1, 1, 0),
                # nn.Conv2d(out_ch//4, 3, 3, 1, 1),
                # nn.Sigmoid()            
            )

            self.denoise_deconv =  nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(deconv_in_dim, out_ch//2, 3, 1, 1),
                nn.BatchNorm2d(out_ch//2),
                nn.LeakyReLU(0.2, True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(out_ch//2, out_ch//4, 3, 1, 1),
                nn.BatchNorm2d(out_ch//4),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_ch//4, 3, 1, 1, 0),
                # nn.Conv2d(out_ch//4, 3, 3, 1, 1),
                # nn.Sigmoid()            
            )


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track_running_stats=False, affine=True),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    @staticmethod
    def skipconnect(x1, x2):
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x, conv1_weights=None, reconstruct=False):
        if conv1_weights == None:
            x = self.conv1(x)
        else:
            conv1_weights = conv1_weights.reshape(x.shape[0], self.conv1_in_dim, 3, 7, 7)
            meta_half = []
            x_half = self.conv1_half(x) 
            x = F.pad(x, pad=(3,3,3,3), mode='reflect')
            for xi, conv1_weight in zip(x, conv1_weights):
                hi = F.conv2d(xi[None], conv1_weight, stride=2, padding=0)
                meta_half.append(hi)
            meta_half = torch.cat(meta_half, dim=0)
            x = torch.cat([x_half, meta_half], dim=1)


        x = self.relu(self.bn1(x))

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_out = self.out_conv(x)

        if self.coarse_only:
            x_coarse = x_out
            x_fine = None
        else:
            x_coarse = x_out[:, :self.coarse_out_ch, :]
            x_fine = x_out[:, -self.fine_out_ch:, :]

        out_dict = {'coarse': x_coarse, 'fine': x_fine}
        if self.auto_encoder and reconstruct:
            if not self.per_level_render:
                x_reconst_coarse = self.reconst_deconv(x_coarse)
                x_denoised_coarse = self.denoise_deconv(x_coarse)

                x_reconst_fine = self.reconst_deconv(x_fine)
                x_denoised_fine = self.denoise_deconv(x_fine)

                out_dict['reconst_signal'] = (x_reconst_coarse + x_reconst_fine) / 2 
                out_dict['denoised_signal'] = (x_denoised_coarse + x_denoised_fine) / 2
                    
            else:

                x_reconst_coarse = self.reconst_deconv(x_coarse)
                x_denoised_coarse = self.denoise_deconv(x_coarse)

                x_reconst_fine = self.reconst_deconv(x_fine)
                x_denoised_fine = self.denoise_deconv(x_fine)

                out_dict['reconst_signal_coarse'] = x_reconst_coarse 
                out_dict['reconst_signal_fine'] = x_reconst_fine 
                
                out_dict['denoised_signal_coarse'] = x_denoised_coarse
                out_dict['denoised_signal_fine'] = x_denoised_fine
                    
                
        return out_dict

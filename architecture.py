import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as  np


def stack_image(image, N=8, pad=2):
    b, c, h, w = image.size()
    assert h % N == 0 and w % 8 == 0
    padded_image = torch.nn.functional.pad(image, (pad, pad, pad, pad), mode='constant', value=0)
    patch_h = h//N
    patch_w = w//N 
    stacked = torch.zeros((b, c*N*N, patch_h + 2*pad, patch_w + 2*pad)).to(image.device)
    for j in range(N):
        for k in range(N):
            row_start = j* patch_h
            row_end = (j+1)* patch_h + 2*pad
            col_start = k* patch_w
            col_end = (k+1)* patch_w + 2*pad
            stacked[:, c * (j * N + k): c * (j * N + k + 1)] = padded_image[:, :, row_start:row_end, col_start:col_end]
    return stacked


def unstack_image(stack_image, total_n_patch):
    b, c, patch_h, patch_w = stack_image.size()
    assert c % total_n_patch == 0
    N = int(total_n_patch ** 0.5)
    output_c = c // total_n_patch
    output = torch.zeros((b, output_c, patch_h * N, patch_w * N)).to(stack_image.device)

    for patch_idx in range(total_n_patch):
        h_idx = patch_idx // N
        w_idx = patch_idx % N
        output[:, :, h_idx * patch_h : (h_idx + 1) * patch_h, w_idx * patch_w : (w_idx + 1) * patch_w] = stack_image[:,patch_idx * output_c: (patch_idx + 1) * output_c]
    return output



class CNN_Encoder(nn.Module):
    def __init__(self, meta_module, meta_half, meta_independent, meta_residual, meta_decoder=False, extra_dims=None, patch_kernel=False):
        super(CNN_Encoder, self).__init__()

        
        self.meta_module = meta_module
        self.meta_half = meta_half
        self.meta_independent = meta_independent
        self.meta_residual = meta_residual
        self.meta_decoder = meta_decoder
        self.channel_mult = 16
        self.patch_kernel = patch_kernel
        self.total_patch_n = 64
        # self.channel_mult = 16

        self.extra_dims = 0 if not meta_decoder else extra_dims

        #convolutions
        if not self.meta_module:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3,
                        out_channels=self.channel_mult*1,
                        kernel_size=5,
                        stride=1,
                        padding=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 5, 2, 2),
                nn.BatchNorm2d(self.channel_mult*2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            if self.meta_half and not self.meta_residual:
                self.conv11 = nn.Conv2d(in_channels=3, out_channels=self.channel_mult*1 // 2, kernel_size=5, stride=1, padding=2)
                self.conv12 = nn.Conv2d(self.channel_mult*1, self.channel_mult*2 // 2, 5, 2, 2)
            self.bn = nn.BatchNorm2d(self.channel_mult*2)
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 5, 2, 2),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 5, 2, 2),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*16, self.channel_mult*16 + self.extra_dims, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*16 + self.extra_dims),
            nn.LeakyReLU(0.2, inplace=True)            
        )


    def indep_half_meta_forward(self, x, conv_weights):
        if self.patch_kernel:
            conv1_weights, conv2_weights = conv_weights.split([3 * self.channel_mult // 2 * 5 * 5, self.channel_mult * self.channel_mult * 2 // 2 * 5 * 5], dim=-1) 

            conv1_weights = conv1_weights.reshape(conv1_weights.shape[0], -1, self.channel_mult // 2, 3, 5, 5)
            conv1_weights = conv1_weights.reshape(conv1_weights.shape[0], -1, 3, 5, 5)

            conv2_weights = conv2_weights.reshape(conv2_weights.shape[0], -1, self.channel_mult * 2 // 2, self.channel_mult, 5, 5)
            conv2_weights = conv2_weights.reshape(conv2_weights.shape[0], -1, self.channel_mult, 5, 5)

            xx = self.conv11(x)
            x_stacked = stack_image(x, N=int(self.total_patch_n ** 0.5), pad=2)

            half = []
            for xi, conv1_weight in zip(x_stacked, conv1_weights):
                hi = F.conv2d(xi[None], conv1_weight, stride=1, padding=0, groups=self.total_patch_n)
                half.append(hi)

            half = torch.cat(half, dim=0)
            half = unstack_image(half, self.total_patch_n)
            x = torch.cat([xx, half], dim=1)
            x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

            xx = self.conv12(x)
            half = []
            x_stacked = stack_image(x, N=int(self.total_patch_n ** 0.5), pad=2)
            for xi, conv2_weight in zip(x_stacked, conv2_weights):
                hi = F.conv2d(xi[None], conv2_weight, stride=2, padding=0, groups=self.total_patch_n)
                half.append(hi)
            half = torch.cat(half, dim=0)
            half = unstack_image(half, self.total_patch_n)
            x = torch.cat([xx, half], dim=1)

            x = self.bn(x)
            x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        else:
            conv1_weights, conv2_weights = conv_weights.split([3 * self.channel_mult // 2 * 5 * 5, self.channel_mult * self.channel_mult * 2 // 2 * 5 * 5], dim=1) 
            conv1_weights = conv1_weights.reshape(conv1_weights.shape[0], self.channel_mult // 2, 3, 5, 5)
            conv2_weights = conv2_weights.reshape(conv2_weights.shape[0], self.channel_mult * 2 // 2, self.channel_mult, 5, 5)
            
            xx = self.conv11(x)
            half = []
            for xi, conv1_weight in zip(x, conv1_weights):
                hi = F.conv2d(xi[None], conv1_weight, stride=1, padding=2)
                half.append(hi)
            half = torch.cat(half, dim=0)
            x = torch.cat([xx, half], dim=1)

            x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

            xx = self.conv12(x)
            half = []
            for xi, conv2_weight in zip(x, conv2_weights):
                hi = F.conv2d(xi[None], conv2_weight, stride=2, padding=2)
                half.append(hi)
            half = torch.cat(half, dim=0)
            x = torch.cat([xx, half], dim=1)

            x = self.bn(x)
            x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        return x

    def forward(self, x, conv_weights=None):
        input_h, input_w = x.shape[-2:]
        xcp = x.clone()
        if self.meta_module:
            x = self.indep_half_meta_forward(x, conv_weights)
        else:
            x = self.conv(x.view(x.shape[0], -1, input_h, input_w))
        x = self.conv2(x)
        x = x.view(-1, self.channel_mult * 16 + self.extra_dims, input_h // 32, input_w // 32)

        return x

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, meta_decoder=False):
        super(CNN_Decoder, self).__init__()
        # self.channel_mult = 8
        self.channel_mult = 16
        self.output_channels = 3

        self.meta_decoder = meta_decoder
        if not meta_decoder:
            self.deconv_0 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.channel_mult * 16, self.channel_mult*4, 3, 1, 1),
                nn.BatchNorm2d(self.channel_mult*4),
                nn.ReLU(True),
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.bn = nn.BatchNorm2d(self.channel_mult*4)
            self.relu = nn.ReLU(True)
            
        self.deconv_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*2, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*2, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*2, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
        )
        self.final_deconv =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*1, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
        )
        self.final_deconv_2 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*1, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
        )
        self.final_deconv_3 =  nn.Sequential(
            nn.Conv2d(self.channel_mult*1, self.output_channels, 3, 1, 1),
            nn.Sigmoid()            
        )
        ## 
        self.reconst_conv =  nn.Sequential(
            nn.Conv2d(self.channel_mult*2, self.output_channels, 1, 1, 0),
            nn.Sigmoid()            
        )
        self.reconst_conv_2 =  nn.Sequential(
            nn.Conv2d(self.channel_mult*1, self.output_channels, 1, 1, 0),
            nn.Sigmoid()            
        )

    def forward(self, x, multiscale=False, conv_weights=None):
        if self.meta_decoder:
            conv_weights = conv_weights.reshape(self.channel_mult * 16, self.channel_mult*4, 3, 3)
            conv_weights = conv_weights.permute(1,0,2,3)
            x = self.upsample(x)
            x = F.conv2d(x, conv_weights, stride=1, padding=1)            
            x = self.bn(x)
            x = self.relu(x)
        else:
            x = self.deconv_0(x)
        x = self.deconv_1(x)
        x_down2 = self.final_deconv(x)

        reconst_down4 = self.reconst_conv(x) if multiscale else None
        reconst_down2 = self.reconst_conv_2(x_down2) if multiscale else None

        reconst_x = self.final_deconv_2(x_down2)
        reconst_x = self.final_deconv_3(reconst_x)
        return [reconst_x, reconst_down2, reconst_down4], x

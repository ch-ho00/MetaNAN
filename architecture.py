import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as  np

TOTAL_PATCH_N = 64 

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
    def __init__(self, meta_module, patch_kernel=False):
        super(CNN_Encoder, self).__init__()

        
        self.meta_module = meta_module
        self.channel_mult = 8
        self.patch_kernel = patch_kernel
        self.total_patch_n = 64
        # self.channel_mult = 16

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
            nn.Conv2d(self.channel_mult*16, self.channel_mult*16, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*16),
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
        if self.meta_module:
            x = self.indep_half_meta_forward(x, conv_weights)
        else:
            x = self.conv(x.view(x.shape[0], -1, input_h, input_w))
        x = self.conv2(x)
        x = x.view(-1, self.channel_mult * 16, input_h // 32, input_w // 32)

        return x

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size):
        super(CNN_Decoder, self).__init__()
        self.input_dim = embedding_size
        self.channel_mult = 8
        # self.channel_mult = 16
        self.output_channels = 3

        self.deconv_0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.channel_mult * 16, self.channel_mult*4, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
        )
            
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

        x = self.deconv_0(x)
        x = self.deconv_1(x)
        x_down2 = self.final_deconv(x)

        reconst_down4 = self.reconst_conv(x) if multiscale else None
        reconst_down2 = self.reconst_conv_2(x_down2) if multiscale else None

        reconst_x = self.final_deconv_2(x_down2)
        reconst_x = self.final_deconv_3(reconst_x)
        return [reconst_x, reconst_down2, reconst_down4], x

#######################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, meta_module=False, patch_kernel=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.meta_module = meta_module
        self.patch_kernel = patch_kernel

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels // (2 if meta_module else 1), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels // (2 if meta_module else 1), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if meta_module:
            self.conv_weights_dim = self.in_channels * 3 * 3 * self.mid_channels // 2 + self.mid_channels * 3 * 3 * self.out_channels // 2
        else:
            self.conv_weights_dim = 0

    def init_weight(self):
        torch.nn.init.constant_(self.double_conv[0].weight, 0)
        torch.nn.init.constant_(self.double_conv[3].weight, 0)

    def forward(self, x, conv_weights=None):
        if self.meta_module:
            if self.patch_kernel:
                conv1_weights, conv2_weights = conv_weights.split([
                    self.in_channels * 3 * 3 * self.mid_channels // 2, 
                    self.mid_channels * 3 * 3 * self.out_channels // 2
                ], dim=-1) 

                conv1_weights = conv1_weights.reshape(conv1_weights.shape[0], -1, self.mid_channels // 2, self.in_channels, 3, 3)
                conv1_weights = conv1_weights.reshape(conv1_weights.shape[0], -1, self.in_channels, 3, 3)

                xx = self.double_conv[0](x)
                x_stacked = stack_image(x, N=int(TOTAL_PATCH_N ** 0.5), pad=1)

                half = []
                for xi, conv1_weight in zip(x_stacked, conv1_weights):
                    hi = F.conv2d(xi[None], conv1_weight, stride=1, padding=0, groups=TOTAL_PATCH_N)
                    half.append(hi)
                half = torch.cat(half, dim=0)
                half = unstack_image(half, TOTAL_PATCH_N)
                x = torch.cat([xx, half], dim=1)

                x = self.double_conv[1](x)
                x = self.double_conv[2](x)

                conv2_weights = conv2_weights.reshape(conv2_weights.shape[0], -1, self.out_channels // 2, self.mid_channels, 3, 3)
                conv2_weights = conv2_weights.reshape(conv2_weights.shape[0], -1, self.mid_channels, 3, 3)

                xx = self.double_conv[3](x)
                x_stacked = stack_image(x, N=int(TOTAL_PATCH_N ** 0.5), pad=1)
                half = []
                for xi, conv2_weight in zip(x_stacked, conv2_weights):
                    hi = F.conv2d(xi[None], conv2_weight, stride=1, padding=0, groups=TOTAL_PATCH_N)
                    half.append(hi)
                half = torch.cat(half, dim=0)
                half = unstack_image(half, TOTAL_PATCH_N)
                x = torch.cat([xx, half], dim=1)

                x = self.double_conv[4](x)
                x = self.double_conv[5](x)
                return x
            else:
                conv1_weights, conv2_weights = conv_weights.split([
                    self.in_channels * 3 * 3 * self.mid_channels // 2,
                    self.mid_channels * 3 * 3 * self.out_channels // 2
                ], dim=1)
                conv1_weights = conv1_weights.reshape(conv1_weights.shape[0], self.mid_channels // 2, self.in_channels, 3, 3)
                conv2_weights = conv2_weights.reshape(conv2_weights.shape[0], self.out_channels // 2, self.mid_channels, 3, 3)

                xx = self.double_conv[0](x)
                half = []
                for xi, conv1_weight in zip(x, conv1_weights):
                    hi = F.conv2d(xi[None], conv1_weight, stride=1, padding=1)
                    half.append(hi)
                half = torch.cat(half, dim=0)
                x = torch.cat([xx, half], dim=1)

                x = self.double_conv[1](x)
                x = self.double_conv[2](x)

                xx = self.double_conv[3](x)
                half = []
                for xi, conv2_weight in zip(x, conv2_weights):
                    hi = F.conv2d(xi[None], conv2_weight, stride=1, padding=1)
                    half.append(hi)
                half = torch.cat(half, dim=0)
                x = torch.cat([xx, half], dim=1)

                x = self.double_conv[4](x)
                x = self.double_conv[5](x)
                return x
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, meta_module=False, patch_kernel=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, meta_module=meta_module, patch_kernel=patch_kernel)
        )

        self.conv_weights_dim = self.maxpool_conv[1].conv_weights_dim

    def init_weight(self):
        self.maxpool_conv[1].init_weight()


    def forward(self, x, conv_weights=None):
        if conv_weights is not None:
            x = self.maxpool_conv[0](x)
            x = self.maxpool_conv[1](x, conv_weights)
            return x
        else:
            return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def init_weight(self):
        self.conv.init_weight()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def init_weight(self):
        torch.nn.init.constant_(self.conv.weight, 0)
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class UNet_Encoder(nn.Module):
    def __init__(self, meta_module, bilinear=False, patch_kernel=False):
        super(UNet_Encoder, self).__init__()
        self.channel_mult = 16
        self.patch_kernel = patch_kernel
        scale = 128 // self.channel_mult

        self.inc = (DoubleConv(3, 64 // scale, meta_module=meta_module, patch_kernel=patch_kernel))
        self.down1 = (Down(64 // scale, 128 // scale, meta_module=meta_module, patch_kernel=patch_kernel))
        self.down2 = (Down(128 // scale, 256 // scale, meta_module=False, patch_kernel=False))
        self.down3 = (Down(256 // scale, 512 // scale, meta_module=False, patch_kernel=False))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512 // scale, 1024 // scale // factor, meta_module=False, patch_kernel=False))

        self.conv_weights_dims = [
            self.inc.conv_weights_dim, 
            self.down1.conv_weights_dim,
            self.down2.conv_weights_dim,
            self.down3.conv_weights_dim,
            self.down4.conv_weights_dim
        ]
        self.conv_weights_dim = sum(self.conv_weights_dims)
        self.init_weight()

    def init_weight(self):
        self.down1.init_weight()
        self.down2.init_weight()
        self.down3.init_weight()
        self.down4.init_weight()

    def forward(self, x, conv_weights=None):
        if conv_weights is not None:
            conv_weights = conv_weights.split(self.conv_weights_dims, dim=-1) 
        else:
            conv_weights = [None] * len(self.conv_weights_dims)

        x1 = self.inc(x, conv_weights[0]) # 1 -> 1
        x2 = self.down1(x1, conv_weights[1]) # 1 -> 2
        x3 = self.down2(x2, conv_weights[2]) # 2 -> 4
        x4 = self.down3(x3, conv_weights[3]) # 4 -> 8
        x5 = self.down4(x4, conv_weights[4]) # 8 -> 16

        return [x1, x2, x3, x4, x5]
    
class UNet_Decoder(nn.Module):
    def __init__(self, bilinear=False):
        super(UNet_Decoder, self).__init__()
        self.channel_mult = 16
        scale = 128 // self.channel_mult

        factor = 2 if bilinear else 1
        self.down4 = (Down(512 // scale, 1024 // scale // factor))
        self.up1 = (Up(1024 // scale, 512 // scale // factor, bilinear))
        self.up2 = (Up(512 // scale, 256 // scale // factor, bilinear))
        self.up3 = (Up(256 // scale, 128 // scale // factor, bilinear))
        self.up4 = (Up(128 // scale, 64 // scale, bilinear))
        self.outc = (OutConv(64 // scale, 3))

        self.reconst_conv =  nn.Sequential(
            nn.Conv2d(256 // scale // factor, 3, 1, 1, 0),
            nn.Sigmoid()            
        )
        self.reconst_conv_2 =  nn.Sequential(
            nn.Conv2d(128 // scale // factor, 3, 1, 1, 0),
            nn.Sigmoid()            
        )
        self.init_weight()

    def init_weight(self):
        self.down4.init_weight()
        self.up1.init_weight()
        self.up2.init_weight()
        self.up3.init_weight()
        self.up4.init_weight()
        self.outc.init_weight()

        torch.nn.init.constant_(self.reconst_conv[0].weight, 0)
        torch.nn.init.constant_(self.reconst_conv[0].bias, 0)

        torch.nn.init.constant_(self.reconst_conv_2[0].weight, 0)
        torch.nn.init.constant_(self.reconst_conv_2[0].bias, 0)

    def forward(self, x, multiscale=False, conv_weights=None):
        x1, x2, x3, x4, x5 = x

        x_down8 = self.up1(x5, x4) # 16 -> 8
        x_down4 = self.up2(x_down8, x3) # 8 -> 4
        x_down2 = self.up3(x_down4, x2) # 4 -> 2
        reconst_x = self.up4(x_down2, x1) # 2 -> 1
        reconst_x = self.outc(reconst_x) # 1 -> 1

        reconst_down4 = self.reconst_conv(x_down4) if multiscale else None
        reconst_down2 = self.reconst_conv_2(x_down2) if multiscale else None

        reconst = [reconst_x, reconst_down2, reconst_down4] if multiscale else reconst_x
        return reconst, x_down4


from inplace_abn import InPlaceABN
#############################################     MVS Net models        ################################################
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))


###################################  feature net  ######################################
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, out_feat, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3 , 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, out_feat, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        x = self.conv2(x) # (B, 32, H//4, W//4)
        x = self.toplayer(x) # (B, 32, H//4, W//4)

        return x


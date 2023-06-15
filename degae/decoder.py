import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='reflect')


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='reflect')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, rand_noise=True):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.down_conv = conv3x3(inplanes, planes, stride) if downsample else None
        self.stride = stride
        self.rand_noise = rand_noise
        self.weight = nn.Parameter(torch.randn(planes) * 0.01)

    def forward(self, x, scale=None, shift=None):
        B = x.shape[0]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if scale != None:
            out = out * scale.view(-1, out.shape[1], 1, 1) + shift.view(-1, out.shape[1], 1, 1)
            if self.rand_noise:
                out = out + (self.weight * torch.randn_like(self.weight)).reshape(1,-1,1,1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.down_conv(x)

        out += identity
        out = self.relu(out)

        return out
    

class DegAE_decoder(nn.Module):

    def __init__(self, img_feat_dim=64, cond_vec_dim=512, rand_noise=False, skip_condition=False):    
        super().__init__()
        self.img_feat_dim = 64
        self.block1 = BasicBlock(img_feat_dim, img_feat_dim, stride=1, downsample=None, rand_noise=rand_noise)
        self.block2 = BasicBlock(img_feat_dim, img_feat_dim, stride=1, downsample=None, rand_noise=rand_noise)
        self.block3 = BasicBlock(img_feat_dim, img_feat_dim, stride=1, downsample=None, rand_noise=rand_noise)
        self.block4 = BasicBlock(img_feat_dim, img_feat_dim, stride=1, downsample=None, rand_noise=rand_noise)

        self.final_conv = nn.Conv2d(img_feat_dim, 3, 1, 1, 0)
        self.skip_condition = skip_condition
        if not self.skip_condition:
            self.cond_scale1 = nn.Linear(cond_vec_dim, img_feat_dim, bias=True)
            self.cond_scale2 = nn.Linear(cond_vec_dim, img_feat_dim,  bias=True)
            self.cond_scale3 = nn.Linear(cond_vec_dim, img_feat_dim, bias=True)
            self.cond_scale4 = nn.Linear(cond_vec_dim, img_feat_dim, bias=True)

            self.cond_shift1 = nn.Linear(cond_vec_dim, img_feat_dim, bias=True)
            self.cond_shift2 = nn.Linear(cond_vec_dim, img_feat_dim, bias=True)
            self.cond_shift3 = nn.Linear(cond_vec_dim, img_feat_dim, bias=True)
            self.cond_shift4 = nn.Linear(cond_vec_dim, img_feat_dim, bias=True)

        
    
    def forward(self, x, degrade_vec=None):

        scale1, shift1 = None, None 
        if degrade_vec != None:
            scale1 = self.cond_scale1(degrade_vec)
            shift1 = self.cond_shift1(degrade_vec)    
        x = self.block1(x, scale1, shift1)
        
        scale2, shift2 = None, None 
        if degrade_vec != None:
            scale2 = self.cond_scale2(degrade_vec)
            shift2 = self.cond_shift2(degrade_vec)
        x = self.block2(x, scale2, shift2)

        scale3, shift3 = None, None 
        if degrade_vec != None:
            scale3 = self.cond_scale3(degrade_vec)
            shift3 = self.cond_shift3(degrade_vec)
        x = self.block3(x, scale3, shift3)
        

        scale4, shift4 = None, None 
        if degrade_vec != None:
            scale4 = self.cond_scale4(degrade_vec)
            shift4 = self.cond_shift4(degrade_vec)
        x = self.block4(x, scale4, shift4)
        x = self.final_conv(x)

        return x
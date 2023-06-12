import math
import os
from typing import Any, cast, Dict, List, Union
import torch.nn as nn
from torch import Tensor as Tensor
import torch 
import torch.nn.functional as F
import sys
sys.path.append('/disk1/chanho/3d/MetaNAN')
from nan.dataloaders.basic_dataset import de_linearize
from .srgan import SRResNet

class DegFeatureExtractor(nn.Module):
    def __init__(
            self, vgg_ckpt
    ) -> None:
        super(DegFeatureExtractor, self).__init__()
        self.vgg = DiscriminatorForVGG(in_channels=3, out_channels=1, channels=64)
        model_weights_path = vgg_ckpt
        checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
        self.vgg.load_state_dict(checkpoint["state_dict"])
        # self.srgan = SRResNet(in_channels=3,
        #                       out_channels=3,
        #                       channels=64,
        #                       num_rcb=16,
        #                       upscale=4)
        # model_weights_path = vgg_ckpt
        # checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
        # self.srgan.load_state_dict(checkpoint["state_dict"])
        # print(self.srgan(torch.randn(1,3,100,100)).shape)
        # self.degrep_conv = nn.Sequential(
        #     nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        # ) 
        self.degrep_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
        ) 

        self.degrep_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
    def forward(self, x, white_level) -> Tensor:
        x = de_linearize(x, white_level).clamp(0,1)
        x = self.vgg(x)
        # x = self.srgan(x)
        x = self.degrep_conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.degrep_fc(x.reshape(-1,512))
        return x

class DiscriminatorForVGG(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
    ) -> None:
        super(DiscriminatorForVGG, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(channels, channels, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, int(2 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(int(2 * channels), int(2 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(2 * channels), int(4 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(int(4 * channels), int(4 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(4 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(int(8 * channels), int(8 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(8 * channels) * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, out_channels),
        )

    def forward(self, x) -> Tensor:
        # Input image size must equal 96
        # assert x.size(2) == 96 and x.size(3) == 96, "Input image size must be is 96x96"
        x = self.features(x)
        if x.isnan().sum() > 0:
            import pdb; pdb.set_trace()

        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)

        return x
    
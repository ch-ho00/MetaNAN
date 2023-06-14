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
            self, vgg_ckpt, train_scratch=True
    ) -> None:
        super(DegFeatureExtractor, self).__init__()

        self.srgan = SRResNet(in_channels=3,
                              out_channels=3,
                              channels=64,
                              num_rcb=16,
                              upscale=4)
        if train_scratch:
            model_weights_path = vgg_ckpt
            checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
            self.srgan.load_state_dict(checkpoint["state_dict"])
        self.degrep_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
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
        # x = self.vgg(x)
        x = self.srgan(x)
        x = self.degrep_conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.degrep_fc(x.reshape(-1,512))
        return x


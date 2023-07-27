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
            self, ckpt_path, train_scratch=True
    ) -> None:
        super(DegFeatureExtractor, self).__init__()

        self.srgan = SRResNet(in_channels=3,
                              out_channels=3,
                              channels=64,
                              num_rcb=16,
                              upscale=4)
        if train_scratch:
            assert ckpt_path != None
            model_weights_path = ckpt_path
            checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
            self.srgan.load_state_dict(checkpoint["state_dict"])

        for d_parameters in self.srgan.parameters():
            d_parameters.requires_grad = False
        self.srgan.eval()

        self.degrep_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        ) 
        if not train_scratch:
            for d_parameters in self.degrep_conv.parameters():
                d_parameters.requires_grad = False
            self.degrep_conv.eval()

        self.degrep_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
        if not train_scratch:
            for d_parameters in self.degrep_fc.parameters():
                d_parameters.requires_grad = False
            self.degrep_fc.eval()

    def forward(self, x, white_level) -> Tensor:
        if white_level != None:
            if white_level.ndim == 2 and white_level.shape[0] == 1:
                white_level = white_level[0].item()
            x = de_linearize(x, white_level) #.clamp(0,1)
        with torch.no_grad():
            x = self.srgan(x)
        x = self.degrep_conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.degrep_fc(x.reshape(-1,512))
        return x


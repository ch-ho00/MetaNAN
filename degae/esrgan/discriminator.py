import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torch.nn.utils import spectral_norm
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

class DiscriminatorUNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            upsample_method: str = "bilinear",
    ) -> None:
        super(DiscriminatorUNet, self).__init__()
        self.upsample_method = upsample_method

        self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1), (1, 1))
        self.down_block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, int(channels * 2), (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 2), int(channels * 4), (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 4), int(channels * 8), (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 8), int(channels * 4), (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 4), int(channels * 2), (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 2), channels, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    def save_model(self, filename, optimizer, scheduler):
        to_save = {'optimizer'  : optimizer.state_dict(),
                   'scheduler'  : scheduler.state_dict(),
                   'model' : de_parallel(self).state_dict()}
        torch.save(to_save, filename)


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)

        # Down-sampling
        down1 = self.down_block1(out1)
        down2 = self.down_block2(down1)
        down3 = self.down_block3(down2)

        # Up-sampling
        down3 = F_torch.interpolate(down3, scale_factor=2, mode="bilinear", align_corners=False)
        up1 = self.up_block1(down3)

        up1 = torch.add(up1, down2)
        up1 = F_torch.interpolate(up1, scale_factor=2, mode="bilinear", align_corners=False)
        up2 = self.up_block2(up1)

        up2 = torch.add(up2, down1)
        up2 = F_torch.interpolate(up2, scale_factor=2, mode="bilinear", align_corners=False)
        up3 = self.up_block3(up2)

        up3 = torch.add(up3, out1)

        out = self.conv2(up3)
        out = self.conv3(out)
        out = self.conv4(out)

        return out

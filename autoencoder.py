import torch
import torch.nn as nn
from torch.nn import functional as F
from architecture import CNN_Encoder, CNN_Decoder
from architecture import UNet_Encoder, UNet_Decoder
from architecture import FeatureNet
from nan.feature_network import ResUNet

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

        self.out_dim = 128
        self.conv0 = UNetConvBlock(3, 64, True, 0.2)
        self.conv1 = UNetConvBlock(64, 128, True, 0.2)
        self.conv2 = UNetConvBlock(128, 64, True, 0.2)
        self.conv3 = UNetConvBlock(64, self.out_dim, True, 0.2)

    def forward(self, x):
        x = self.conv0(x) # (B, 32, H//2, W//2)
        x = self.conv1(x) # (B, 64, H//4, W//4)
        x = self.conv2(x) # (B, 128, H//8, W//8)
        x = self.conv3(x) # (B, 256, H//16, W//16)
        x = F.adaptive_avg_pool2d(x, (8, 8))
        return x

class ConvWeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, patch_kernel=False):
      super(ConvWeightGenerator,self).__init__()
      self.in_dim = in_dim
      self.out_dim = out_dim
      self.patch_kernel = patch_kernel

      self.transform = nn.Sequential(
        nn.Linear(self.in_dim, 1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024,1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024,self.out_dim)
      )

    def forward(self,noise_vec):
      if self.patch_kernel:
        noise_vec = noise_vec.reshape(noise_vec.shape[0], self.in_dim, -1).permute(0,2,1)
      weights = self.transform(noise_vec)
      return weights 

class AutoEncoder(nn.Module):
    def __init__(self, args, meta_module, patch_kernel=False):
        super(AutoEncoder, self).__init__()
        self.decoder = UNet_Decoder(bilinear=False)
        self.encoder = UNet_Encoder(meta_module, bilinear=False, patch_kernel=patch_kernel)        
        # self.feature_net = FeatureNet(self.decoder.up2.conv.out_channels)
        self.feature_net = ResUNet(encoder='resnet18', coarse_out_ch=args.coarse_feat_dim,
                                fine_out_ch=args.fine_feat_dim,
                                coarse_only=args.coarse_only)
        # self.decoder = CNN_Decoder()
        # self.encoder = CNN_Encoder(meta_module, patch_kernel=patch_kernel)

    def forward(self, x, conv_weights, multiscale=False):
        z = self.encoder(x, conv_weights)
        reconst_x, _ = self.decoder(z, multiscale=multiscale)
        #print(torch.max(reconst_x), torch.min(reconst_x), torch.mean(reconst_x), reconst_x.shape)
        reconst_x = reconst_x + x
        feature = self.feature_net(reconst_x)
        return feature, reconst_x



import torch
import torch.nn as nn
from architecture import CNN_Encoder, CNN_Decoder
from torch.nn import functional as F



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

        self.out_dim = 512
        self.conv0 = UNetConvBlock(3, 64, True, 0.2)
        self.conv1 = UNetConvBlock(64, 128, True, 0.2)
        self.conv2 = UNetConvBlock(128, 256, True, 0.2)
        self.conv3 = UNetConvBlock(256, self.out_dim, True, 0.2)

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
    def __init__(self, output_size, meta_module, meta_half, meta_independent, meta_residual, meta_decoder=False, patch_kernel=False):
        super(AutoEncoder, self).__init__()
        
        self.decoder = CNN_Decoder(output_size, meta_decoder)
        extra_dims = 0 if not meta_decoder else self.decoder.channel_mult * 16
        self.encoder = CNN_Encoder(output_size, meta_module, meta_half, meta_independent, meta_residual, meta_decoder, extra_dims, patch_kernel=patch_kernel)
        
        self.meta_decoder = meta_decoder
        if meta_decoder:
            out_dim = self.decoder.channel_mult * 4 * 3 * 3
            _, h, w = input_size
            in_dim = (h // 32) * (w // 32) * 3
            self.weight_generator = ConvWeightGenerator(in_dim=in_dim, out_dim=out_dim)

    def forward(self, x, conv_weights, multiscale=False):
        z = self.encoder(x, conv_weights)
        conv_weights = None
        if self.meta_decoder:
            z, latent_features = torch.split(z, [z.shape[1] - self.encoder.extra_dims, self.encoder.extra_dims], dim=1)            
            latent_features = latent_features.permute(1,0,2,3).reshape(self.encoder.extra_dims, -1)
            conv_weights = self.weight_generator(latent_features)
        reconst_x, feature = self.decoder(z, multiscale=multiscale, conv_weights=conv_weights)
        return feature, reconst_x


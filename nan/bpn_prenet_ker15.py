import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from nan.dcn import DeformableConv2d

# BPN basic block: SingleConv
class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output

# BPN basic block: SingleConv
class SingleDeformConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleDeformConv, self).__init__()
        self.conv = nn.Sequential(
            DeformableConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output

class GroupSingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups):
        super(GroupSingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, groups=groups,
                      stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output

# BPN basic block: DownBlock
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output


# BPN basic block: UpBlock
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output



# BPN basic block: CutEdgeConv
# Used to cut the redundant edge of a tensor after 2*2 Conv2d with valid padding
class CutEdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CutEdgeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2,
                      stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output


# BPN basic block: KernelConv
# Used to predict clean image burst via local convolution
class KernelConv(nn.Module):
    def __init__(self, kernel_size=15):
        super(KernelConv, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, data, kernels):
        """
        compute the pred image according to core and frames
        :param data: [batch_size, burst_length, color_channel, height, width]
        :param kernels: [batch_size, burst_length, kernel_size ** 2, color_channel, height, width]
        :return: pred_burst and pred
        """
        if len(data.size()) == 5:
            batch_size, burst_length, color_channel, height, width = data.size()
        else:
            batch_size, burst_length, height, width = data.size()
            color_channel = 1
            data = data.view(batch_size, burst_length, color_channel, height,
                             width)

        img_stack = []
        kernel_size = self.kernel_size
        data_pad = F.pad(data,
                         [kernel_size // 2, kernel_size // 2, kernel_size // 2,
                          kernel_size // 2])
        for i in range(kernel_size):
            for j in range(kernel_size):
                img_stack.append(data_pad[..., i:i + height, j:j + width])
        img_stack = torch.stack(img_stack, dim=2)
        pixel_patch = kernels.mul(img_stack)
        pred_burst = torch.sum(pixel_patch, dim=2, keepdim=False)

        return pred_burst, pixel_patch


class BPN(nn.Module):
    def __init__(self, color=True, burst_length=1, blind_est=True, kernel_size=15, basis_size=64, upMode='bilinear', n_latent_layers=1, channel_upfactor=1, lat_kernel_dim=32, patch_size=5):
        super(BPN, self).__init__()
        self.blind_est = blind_est
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.upMode = upMode
        self.color_channel = 1 #3 if color else 1
        self.burst_length = burst_length   
        self.in_channel = 3 * self.burst_length
        self.n_latent_layers = n_latent_layers
        self.color_channel = self.color_channel 
        factor = 1

        self.skip_connect = True
        self.coeff_channel = self.basis_size * self.n_latent_layers
        self.basis_channel = self.color_channel * self.basis_size * self.n_latent_layers # self.burst_length * 

        # Layer definition in each block
        # Encoder
        self.deconv_channels = [64, 128, 256]
        self.initial_conv = SingleConv(self.in_channel, self.deconv_channels[0])
        self.down_conv1 = DownBlock(self.deconv_channels[0], self.deconv_channels[1])
        self.down_conv2 = DownBlock(self.deconv_channels[1], self.deconv_channels[1])
        self.down_conv3 = DownBlock(self.deconv_channels[1], self.deconv_channels[2])
        self.features_conv1 = SingleConv(self.deconv_channels[2], int(self.deconv_channels[2]  * channel_upfactor))
        # Decoder for coefficients
        self.up_coeff_conv1 = UpBlock(self.deconv_channels[2] + self.deconv_channels[2], self.deconv_channels[1])
        self.up_coeff_conv2 = UpBlock(self.deconv_channels[1] + self.deconv_channels[1], self.deconv_channels[0])
        self.up_coeff_conv3 = UpBlock(self.deconv_channels[1] + self.deconv_channels[0], self.deconv_channels[0])
        if self.n_latent_layers > 1:
            self.coeff_conv1 = SingleDeformConv(self.deconv_channels[0], self.deconv_channels[0])
            self.coeff_conv2 = SingleDeformConv(self.deconv_channels[0], self.deconv_channels[0])
            # self.coeff_conv3 = SingleDeformConv(self.deconv_channels[0], self.coeff_channel)        
            self.coeff_conv3 = GroupSingleConv(self.deconv_channels[0], self.coeff_channel     , groups=self.n_latent_layers)
        else:
            self.coeff_conv1 = SingleConv(self.deconv_channels[0], self.deconv_channels[0])
            self.coeff_conv2 = SingleConv(self.deconv_channels[0], self.deconv_channels[0])
            self.coeff_conv3 = SingleConv(self.deconv_channels[0], self.coeff_channel)
        self.out_coeff = nn.Softmax(dim=1)

        # # Decoder for basis
        self.up_basis_conv1 = UpBlock(self.deconv_channels[2] + self.deconv_channels[2], self.deconv_channels[1])
        self.up_basis_conv2 = UpBlock(self.deconv_channels[1] + self.deconv_channels[1], self.deconv_channels[0])
        self.up_basis_conv3 = UpBlock(self.deconv_channels[1] + self.deconv_channels[0], self.deconv_channels[0])
        self.up_basis_conv4 = UpBlock(self.deconv_channels[0] + self.deconv_channels[0], self.deconv_channels[0])
        self.basis_conv1 = CutEdgeConv(self.deconv_channels[0], self.deconv_channels[0])
        self.basis_conv2 = SingleConv(self.deconv_channels[0], self.deconv_channels[0])
        if self.n_latent_layers > 1:
            self.basis_conv3 = GroupSingleConv(self.deconv_channels[0], self.basis_channel     , groups=self.n_latent_layers)
        else:
            self.basis_conv3 = SingleConv(self.deconv_channels[0], self.basis_channel)
        self.out_basis = nn.Softmax(dim=1)


        # Predict clean images by using local convolutions with kernels
        self.patch_size = patch_size
        self.kernel_conv = KernelConv(self.kernel_size)        
        self.lat_kernel_dim = lat_kernel_dim
        self.kernel_encoder = nn.Sequential(
            nn.Conv2d(self.color_channel * self.kernel_size ** 2, lat_kernel_dim, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Conv2d(self.kernel_size ** 2, 64, kernel_size=1, stride=1, padding=0),
            # nn.Conv2d(64, lat_kernel_dim, kernel_size=1, stride=1, padding=0)
        )         
        self.kernel_decoder = nn.Sequential(
            nn.Conv2d(lat_kernel_dim, self.color_channel * self.kernel_size ** 2, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Conv2d(lat_kernel_dim, 64, kernel_size=1, stride=1, padding=0),
            # nn.Conv2d(64, self.kernel_size ** 2, kernel_size=1, stride=1, padding=0)
        )                 
        if self.n_latent_layers > 1:
            self.offset_conv = nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ) 
            self.offset_fc1 =  nn.Sequential(
                nn.Linear(256, 128),
                nn.Linear(128, 64),
            )
            self.offset_fc2  = nn.Linear(64, 6)

            for module in self.offset_conv.modules():
                if isinstance(module, nn.Conv2d):
                    init.normal_(module.weight, mean=0, std=1e-5)
                    if module.bias is not None:
                        init.constant_(module.bias, 0)

            self.offset_attn = True
            view_att_nhead = 5
            self.views_attention = MultiHeadAttention(view_att_nhead, self.deconv_channels[2], 7, 8)
        else:
            self.offset_attn = False
        # Model weights initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias != None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias != None:
                nn.init.constant_(m.bias.data, 0.0)

    @staticmethod
    def pad_before_cat(x1, x2):
        """Prevent the image dimensions in the encoder and the decoder from
        being different due to the odd image dimension, which will lead to
        skip concatenation failure."""
        diffY = x1.size()[-2] - x2.size()[-2]
        diffX = x1.size()[-1] - x2.size()[-1]
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x2

    @staticmethod
    def pool_before_cat(data, tosize=2):
        """In the decoder for basis, the features are pooled to 1*1 size and
        then enlarged by replication before skip concatenation."""
        if isinstance(tosize, int):
            height = tosize
            width = tosize
        elif isinstance(tosize, list or tuple) and len(tosize) == 2:
            height, width = tosize
        else:
            raise TypeError(
                "Type error on the parameter 'tosize' that denotes the size "
                "of target tensor. Expect to get an int, or a list/tuple with "
                "the length of 2, but got a {}.".format(
                    type(tosize)))
        pooled_data = F.adaptive_avg_pool2d(data, (1, 1))
        return pooled_data.repeat(1, 1, height, width)

    @staticmethod
    def kernel_predict(coeff, basis, batch_size, burst_length, kernel_size,
                       color_channel):
        """
        return size: (batch_size, burst_length, kernel_size ** 2, color_channel, height, width)
        """
        kernels = torch.einsum('ijklmn,ijop->iklmnop', [basis, coeff]).view(
            batch_size, burst_length, kernel_size ** 2, color_channel,
            coeff.size(-2), coeff.size(-1))

        if color_channel == 1:
            kernels = kernels.repeat(1,1,1,3,1,1)

        return kernels

    # forward propagation
    def forward(self, data_with_est, data):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        # input
        initial_conv = self.initial_conv(data_with_est)
        if self.n_latent_layers > 1:
            offset_feats = self.offset_conv(data_with_est)
            offset_feats = offset_feats.permute(0, 2, 3, 1)
            offset_feats = self.offset_fc1(offset_feats)
            pred_offset  = self.offset_fc2(offset_feats)

        # down sampling
        down_conv1 = self.down_conv1(initial_conv)
        down_conv2 = self.down_conv2(
            F.max_pool2d(down_conv1, kernel_size=2, stride=2))
        down_conv3 = self.down_conv3(
            F.max_pool2d(down_conv2, kernel_size=2, stride=2))
        features = self.features_conv1(
            F.max_pool2d(down_conv3, kernel_size=2, stride=2))

        if self.offset_attn:
            reshaped_feats = features.permute(0,2,3,1)
            attn_feats, _ = self.views_attention(offset_feats, reshaped_feats, reshaped_feats)
            features = attn_feats.permute(0,3,1,2)

        # up sampling with skip connection, for coefficients
        up_coeff_conv1 = self.up_coeff_conv1(torch.cat([down_conv3,
                                                        self.pad_before_cat(
                                                            down_conv3,
                                                            F.interpolate(
                                                                features,
                                                                scale_factor=2,
                                                                mode=self.upMode))],
                                                       dim=1))
        up_basis_conv1 = self.up_basis_conv1(torch.cat([self.pool_before_cat(
            down_conv3, tosize=int((self.kernel_size + 1) / 8)), F.interpolate(
            F.adaptive_avg_pool2d(features, (1, 1)), scale_factor=2,
            mode=self.upMode)], dim=1))
        del down_conv3

        up_coeff_conv2 = self.up_coeff_conv2(torch.cat([down_conv2,
                                                        self.pad_before_cat(
                                                            down_conv2,
                                                            F.interpolate(
                                                                up_coeff_conv1,
                                                                scale_factor=2,
                                                                mode=self.upMode))],
                                                       dim=1))
        up_basis_conv2 = self.up_basis_conv2(torch.cat([self.pool_before_cat(
            down_conv2, tosize=int((self.kernel_size + 1) / 4)), F.interpolate(
            up_basis_conv1, scale_factor=2, mode=self.upMode)], dim=1))
        del down_conv2, up_coeff_conv1, up_basis_conv1

        up_coeff_conv3 = self.up_coeff_conv3(torch.cat([down_conv1,
                                                        self.pad_before_cat(
                                                            down_conv1,
                                                            F.interpolate(
                                                                up_coeff_conv2,
                                                                scale_factor=2,
                                                                mode=self.upMode))],
                                                       dim=1))
        up_basis_conv3 = self.up_basis_conv3(torch.cat([self.pool_before_cat(
            down_conv1, tosize=int((self.kernel_size + 1) / 2)), F.interpolate(
            up_basis_conv2, scale_factor=2, mode=self.upMode)], dim=1))
        del down_conv1, up_basis_conv2, up_coeff_conv2
        coeff1 = self.coeff_conv1(up_coeff_conv3)
        del up_coeff_conv3
        coeff2 = self.coeff_conv2(coeff1)
        del coeff1
        coeff3 = self.coeff_conv3(coeff2)
        del coeff2

        if self.n_latent_layers > 1:
            coeffs = []
            for img_idx in range(self.n_latent_layers):
                coeffs.append(self.out_coeff(coeff3[:,self.basis_size * img_idx: self.basis_size * (img_idx + 1)]))
        else:
            coeff = self.out_coeff(coeff3)
        del coeff3

        up_basis_conv4 = self.up_basis_conv4(torch.cat([self.pool_before_cat(
            initial_conv, tosize=int((self.kernel_size + 1) / 1)), F.interpolate(
            up_basis_conv3, scale_factor=2, mode=self.upMode)], dim=1))
        del initial_conv, up_basis_conv3
        basis1 = self.basis_conv1(up_basis_conv4)
        del up_basis_conv4
        basis2 = self.basis_conv2(basis1)
        del basis1
        basis3 = self.basis_conv3(basis2).view(basis2.size(0),
                                               self.basis_size * self.n_latent_layers,
                                               1, #self.burst_length,
                                               self.color_channel,
                                               self.kernel_size,
                                               self.kernel_size)
        del basis2

        if self.n_latent_layers > 1:
            pred_imgs = []
            nchannels = self.basis_size
            for img_idx in range(self.n_latent_layers):
                img_basis = self.out_basis(basis3[:,nchannels * img_idx: nchannels * (img_idx + 1)])
                kernels = self.kernel_predict(coeffs[img_idx], img_basis,
                                            coeffs[img_idx].size(0), 1, self.kernel_size, # self.burst_length
                                            self.color_channel)
                pred_burst = self.kernel_conv(data, kernels)        
                pred_burst = torch.mean(pred_burst, dim=1, keepdim=False)
                pred_imgs.append(pred_burst)
            pred_imgs = torch.stack(pred_imgs, dim=1)
            return pred_imgs, pred_offset.mean(1).mean(1)
        else:
            basis = self.out_basis(basis3)
        del basis3
        # kernel prediction
        kernels = self.kernel_predict(coeff, basis, coeff.size(0),
                                    1, self.kernel_size, # self.burst_length
                                    self.color_channel)
        torch.cuda.empty_cache()
        del coeff, basis
        # clean burst prediction
        pred_burst, pixel_patch = self.kernel_conv(data, kernels)

        N, _, H, W = data_with_est.shape
        # pixel_patch = pixel_patch.reshape(N, self.kernel_size, self.kernel_size, 3, H, W).permute(0,4,5,3,1,2).reshape(-1, 3, self.kernel_size, self.kernel_size)
        # pixel_patch = F.interpolate(pixel_patch, (self.patch_size, self.patch_size))
        # pixel_patch = pixel_patch.reshape(N, H, W, 3, self.patch_size, self.patch_size).permute(0,4,5,3,1,2)
        # pixel_patch = pixel_patch.reshape(N, -1, H, W)

        kernels = kernels.squeeze()[..., 0, :, :] if self.color_channel == 1 else kernels.squeeze().reshape(N, -1, H, W)
        latent_kernel = self.kernel_encoder(kernels)
        reconst_kernels = self.kernel_decoder(latent_kernel)
        ker_reconst_loss = F.l1_loss(reconst_kernels, kernels.detach())

        return pred_burst[:,0], latent_kernel, ker_reconst_loss #, pixel_patch


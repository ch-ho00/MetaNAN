import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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


# BPN basic block: UpBlock
class GroupUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups):
        super(GroupUpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, groups=groups,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, groups=groups,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, groups=groups,
                      stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, data):
        output = self.conv(data)
        return output

class GroupCutEdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups):
        super(GroupCutEdgeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, groups=groups,
                      stride=1, padding=0),
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
        pred_burst = torch.sum(kernels.mul(img_stack), dim=2, keepdim=False)

        return pred_burst


class BPN(nn.Module):
    def __init__(self, color=True, burst_length=1, blind_est=True,
                 kernel_size=7, basis_size=64, upMode='bilinear', bpn_per_img=True, n_latent_layers=None, channel_upfactor=1, group_conv=False, skip_connect=True):
        super(BPN, self).__init__()
        self.blind_est = blind_est
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.upMode = upMode
        self.color_channel = 3 if color else 1
        self.burst_length = burst_length   
        self.in_channel = self.color_channel * self.burst_length
        self.n_latent_layers = n_latent_layers if n_latent_layers != None else 1
        self.color_channel = self.color_channel 
        factor = 1
        self.coeff_channel = self.basis_size * self.n_latent_layers
        self.basis_channel = self.color_channel * self.burst_length * self.basis_size * self.n_latent_layers

        self.skip_connect = skip_connect
        # Layer definition in each block
        # Encoder
        if self.n_latent_layers > 1 and group_conv:
            assert channel_upfactor != None
            self.decode_channels = [( (64 // factor) // self.n_latent_layers +  1) * self.n_latent_layers, ((128 // factor) // self.n_latent_layers +  1) * self.n_latent_layers]

            self.initial_conv = SingleConv(self.in_channel, self.decode_channels[0])
            self.down_conv1 = DownBlock(self.decode_channels[0] , self.decode_channels[0])
            self.down_conv2 = DownBlock(self.decode_channels[0] , self.decode_channels[0])
            self.features_conv1 = GroupSingleConv(self.decode_channels[0], self.decode_channels[1] * channel_upfactor, self.n_latent_layers)
            # Decoder for coeff
            self.up_coeff_conv1 = GroupUpBlock((self.decode_channels[1] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[1]  * channel_upfactor, groups=self.n_latent_layers)
            self.up_coeff_conv2 = GroupUpBlock((self.decode_channels[1] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[0]  * channel_upfactor, groups=self.n_latent_layers)
            self.up_coeff_conv3 = GroupUpBlock((self.decode_channels[0] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[0]  * channel_upfactor, groups=self.n_latent_layers)
            self.coeff_conv1 = GroupSingleConv(self.decode_channels[0] * channel_upfactor, self.decode_channels[0] * channel_upfactor, groups=self.n_latent_layers)
            self.coeff_conv3 = GroupSingleConv(self.decode_channels[0] * channel_upfactor, self.coeff_channel, groups=self.n_latent_layers)

            # # Decoder for basis
            self.up_basis_conv1 = GroupUpBlock((self.decode_channels[1] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[1]  * channel_upfactor, groups=self.n_latent_layers)
            self.up_basis_conv2 = GroupUpBlock((self.decode_channels[1] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[0]  * channel_upfactor, groups=self.n_latent_layers)
            self.up_basis_conv3 = GroupUpBlock((self.decode_channels[0] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[0]  * channel_upfactor, groups=self.n_latent_layers)
            self.basis_conv1 = GroupCutEdgeConv(self.decode_channels[0] * channel_upfactor, self.decode_channels[0] * channel_upfactor, groups=self.n_latent_layers)
            self.basis_conv3 = GroupSingleConv( self.decode_channels[0] * channel_upfactor , self.basis_channel , groups=self.n_latent_layers)

            # Decoder for basis
            # self.up_basis_conv1 = UpBlock((self.decode_channels[1] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[1]  * channel_upfactor)
            # self.up_basis_conv2 = UpBlock((self.decode_channels[1] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[0]  * channel_upfactor)
            # self.up_basis_conv3 = UpBlock((self.decode_channels[0] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[0]  * channel_upfactor)
            # self.basis_conv1 = CutEdgeConv(self.decode_channels[0] * channel_upfactor, self.decode_channels[0] * channel_upfactor) 
            # self.basis_conv3 = SingleConv( self.decode_channels[0] * channel_upfactor , self.basis_channel)

        else:
            self.decode_channels = [64 // factor, 128 // factor]
            self.initial_conv = SingleConv(self.in_channel, self.decode_channels[0])
            self.down_conv1 = DownBlock(self.decode_channels[0] , self.decode_channels[0])
            self.down_conv2 = DownBlock(self.decode_channels[0] , self.decode_channels[0])
            self.features_conv1 = SingleConv(self.decode_channels[0], self.decode_channels[1] * channel_upfactor)
            # Decoder for coeff
            self.up_coeff_conv1 = UpBlock((self.decode_channels[1] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[1]  * channel_upfactor)
            self.up_coeff_conv2 = UpBlock((self.decode_channels[1] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[0]  * channel_upfactor)
            self.up_coeff_conv3 = UpBlock((self.decode_channels[0] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[0]  * channel_upfactor)
            self.coeff_conv1 = SingleConv(self.decode_channels[0] * channel_upfactor, self.decode_channels[0] * channel_upfactor)
            self.coeff_conv3 = SingleConv(self.decode_channels[0] * channel_upfactor, self.coeff_channel)

            # # Decoder for basis
            self.up_basis_conv1 = UpBlock((self.decode_channels[1] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[1]  * channel_upfactor)
            self.up_basis_conv2 = UpBlock((self.decode_channels[1] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[0]  * channel_upfactor)
            self.up_basis_conv3 = UpBlock((self.decode_channels[0] * channel_upfactor  + (self.decode_channels[0] if self.skip_connect else 0)),  self.decode_channels[0]  * channel_upfactor)
            self.basis_conv1 = CutEdgeConv(self.decode_channels[0] * channel_upfactor, self.decode_channels[0] * channel_upfactor)
            self.basis_conv3 = SingleConv( self.decode_channels[0] * channel_upfactor , self.basis_channel )

        self.out_coeff = nn.Softmax(dim=1)
        self.out_basis = nn.Softmax(dim=1)


        # Predict clean images by using local convolutions with kernels
        self.kernel_conv = KernelConv(self.kernel_size)

        # Model weights initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
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

        # down sampling
        down_conv1 = self.down_conv1(
            F.max_pool2d(initial_conv, kernel_size=2, stride=2))
        down_conv2 = self.down_conv2(
            F.max_pool2d(down_conv1, kernel_size=2, stride=2))
        features = self.features_conv1(
            F.max_pool2d(down_conv2, kernel_size=2, stride=2))

        # up sampling with skip connection, for coefficients
        if self.skip_connect:
            up_coeff_conv1 = self.up_coeff_conv1(torch.cat([down_conv2,
                                                            self.pad_before_cat(
                                                                down_conv2,
                                                                F.interpolate(
                                                                    features,
                                                                    scale_factor=2,
                                                                    mode=self.upMode))],
                                                        dim=1))
            up_coeff_conv2 = self.up_coeff_conv2(torch.cat([down_conv1,
                                                            self.pad_before_cat(
                                                                down_conv1,
                                                                F.interpolate(
                                                                    up_coeff_conv1,
                                                                    scale_factor=2,
                                                                    mode=self.upMode))],
                                                        dim=1))
            del up_coeff_conv1
            up_coeff_conv3 = self.up_coeff_conv3(torch.cat([initial_conv,
                                                            self.pad_before_cat(
                                                                initial_conv,
                                                                F.interpolate(
                                                                    up_coeff_conv2,
                                                                    scale_factor=2,
                                                                    mode=self.upMode))],
                                                        dim=1))
        else:
            up_coeff_conv1 = self.up_coeff_conv1(self.pad_before_cat(
                                                                down_conv2,
                                                                F.interpolate(
                                                                    features, 
                                                                    scale_factor=2, 
                                                                    mode=self.upMode)))
            up_coeff_conv2 = self.up_coeff_conv2(self.pad_before_cat(
                                                                down_conv1, 
                                                                F.interpolate(
                                                                    up_coeff_conv1, 
                                                                    scale_factor=2, 
                                                                    mode=self.upMode)))
            del up_coeff_conv1
            up_coeff_conv3 = self.up_coeff_conv3(self.pad_before_cat(
                                                                    initial_conv,
                                                                    F.interpolate(
                                                                        up_coeff_conv2, 
                                                                        scale_factor=2, 
                                                                        mode=self.upMode)))
        del up_coeff_conv2
        coeff1 = self.coeff_conv1(up_coeff_conv3)
        del up_coeff_conv3
        coeff3 = self.coeff_conv3(coeff1)
        del coeff1 

        if self.n_latent_layers > 1:
            coeffs = []
            for img_idx in range(self.n_latent_layers):
                coeffs.append(self.out_coeff(coeff3[:,self.basis_size * img_idx: self.basis_size * (img_idx + 1)]))
        else:
            coeff = self.out_coeff(coeff3)
        del coeff3

        
        # up sampling with pooled-skip connection, for basis
        if self.skip_connect:
            up_basis_conv1 = self.up_basis_conv1(torch.cat([self.pool_before_cat(
                initial_conv, tosize=int((self.kernel_size + 1) / 4)), F.interpolate(
                F.adaptive_avg_pool2d(features, (1, 1)), scale_factor=2,
                mode=self.upMode)], dim=1))
            del initial_conv
            up_basis_conv2 = self.up_basis_conv2(torch.cat([self.pool_before_cat(
                down_conv2, tosize=int((self.kernel_size + 1) / 2)), F.interpolate(
                up_basis_conv1, scale_factor=2, mode=self.upMode)], dim=1))
            del up_basis_conv1, down_conv2
            up_basis_conv3 = self.up_basis_conv3(torch.cat([self.pool_before_cat(
                down_conv1, tosize=int((self.kernel_size + 1) / 1)), F.interpolate(
                up_basis_conv2, scale_factor=2, mode=self.upMode)], dim=1))
            del up_basis_conv2, down_conv1
        else:
            up_basis_conv1 = self.up_basis_conv1(F.adaptive_avg_pool2d(features, (int((self.kernel_size + 1) / 4), int((self.kernel_size + 1) / 4))))
            del initial_conv
            up_basis_conv2 = self.up_basis_conv2(F.interpolate(up_basis_conv1, scale_factor=2, mode=self.upMode))
            del up_basis_conv1, down_conv2
            up_basis_conv3 = self.up_basis_conv3(F.interpolate(up_basis_conv2, scale_factor=2, mode=self.upMode))
            del up_basis_conv2, down_conv1        

        basis1 = self.basis_conv1(up_basis_conv3)
        del up_basis_conv3
        basis3 = self.basis_conv3(basis1).view(basis1.size(0),
                                               self.basis_size * self.n_latent_layers,
                                               self.burst_length,
                                               self.color_channel,
                                               self.kernel_size,
                                               self.kernel_size)
        del basis1
        if self.n_latent_layers > 1:
            # basis = self.out_basis(basis3)
            pred_imgs = []
            nchannels = self.basis_size
            for img_idx in range(self.n_latent_layers):
                img_basis = self.out_basis(basis3[:,nchannels * img_idx: nchannels * (img_idx + 1)])
                kernels = self.kernel_predict(coeffs[img_idx], img_basis,
                                            coeffs[img_idx].size(0), self.burst_length, self.kernel_size,
                                            self.color_channel)
                pred_burst = self.kernel_conv(data, kernels)        
                pred_burst = torch.mean(pred_burst, dim=1, keepdim=False)
                pred_imgs.append(pred_burst)
            pred_imgs = torch.stack(pred_imgs, dim=1)
            torch.cuda.empty_cache()
            return pred_imgs, features
        else:
            basis = self.out_basis(basis3)
        del basis3
        # kernel prediction
        kernels = self.kernel_predict(coeff, basis, coeff.size(0),
                                    self.burst_length, self.kernel_size,
                                    self.color_channel)
        del coeff, basis
        # clean burst prediction
        pred_burst = self.kernel_conv(data, kernels)
        del kernels
        torch.cuda.empty_cache()

        return pred_burst[:,0], features


import torch.nn.init as init
class DeblurBPN(nn.Module):
    def __init__(self, n_latent_layers, burst_length, group_conv, channel_upfactor, skip_connect):
        super(DeblurBPN, self).__init__()

        self.bpn = BPN(bpn_per_img=True, n_latent_layers=n_latent_layers, basis_size=64, burst_length=burst_length, channel_upfactor=channel_upfactor, group_conv=group_conv, skip_connect=skip_connect)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(self.bpn.decode_channels[1] * (channel_upfactor if n_latent_layers > 1 else 1), 64, kernel_size=3, dilation=1, stride=2, padding=0),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, dilation=1, stride=2, padding=0),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, dilation=1, stride=2, padding=0),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, dilation=1, stride=1, padding=0),
            nn.Conv2d(16, 6, kernel_size=1, dilation=1, stride=1, padding=0),
        )

        for module in self.offset_conv.modules():
            if isinstance(module, nn.Conv2d):
                init.normal_(module.weight, mean=0, std=1e-6)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

        # self.offset_fc = nn.Sequential(
        #     nn.Linear(16 * 9, 64),
        #     nn.ELU(inplace=True),
        #     nn.Linear(64, 16),
        #     nn.ELU(inplace=True),
        #     nn.Linear(16, 6),
        #     nn.Sigmoid()
        #     # nn.Tanh()
        # )
        
        # for m in self.offset_fc.modules():
        #     if isinstance(m, nn.Linear):
        #         init.normal_(m.weight, mean=0, std=1e-3)
        #         init.normal_(m.bias, mean=0, std=1e-3)

    def forward(self, input_imgs):
        '''
        Input
            (B,V,3,H,W)
        Output
            (B, n_latent_layers, 3, H, W)
            (B,6)
        '''
        pred_latent_imgs, feature = self.bpn(input_imgs, input_imgs[:,None, :3])
        pred_offset = self.offset_conv(feature).mean(-1).mean(-1)

        return pred_latent_imgs, pred_offset

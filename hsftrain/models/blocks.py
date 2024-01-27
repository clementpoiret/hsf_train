from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F

from hsftrain.models.layers import SwitchNorm3d


def conv3d(in_channels, out_channels, kernel_size, padding, bias, dilation=1):
    return nn.Conv3d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias,
                     dilation=dilation)


def create_conv(in_channels,
                out_channels,
                kernel_size,
                order,
                padding,
                dilation=1):
    """
    Create a list of modules with together constitute a single conv layer with
    non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcg' -> batchnorm + conv + GELU
        dilation (int or tuple): Dilation factor to create dilated conv, and
        increase the recceptive field

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[
        0] not in 'rleg', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(
                ('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'g':
            modules.append(('GELU', nn.GELU()))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm
            bias = not 'b' in order
            modules.append(('conv',
                            conv3d(in_channels,
                                   out_channels,
                                   kernel_size,
                                   bias,
                                   padding=padding,
                                   dilation=dilation)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))

        elif char == 's':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(
                    ('SwitchNorm3d', SwitchNorm3d(in_channels, using_bn=False)))
            else:
                modules.append(
                    ('SwitchNorm3d', SwitchNorm3d(out_channels,
                                                  using_bn=False)))

        elif char == 'i':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(
                    ('InstanceNorm3d', nn.InstanceNorm3d(in_channels)))
            else:
                modules.append(
                    ('InstanceNorm3d', nn.InstanceNorm3d(out_channels)))

        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 's', 'i']"
            )

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and
    optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        dilation (int or tuple): Dilation factor to create dilated conv, and
        increase the recceptive field
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 order='gcr',
                 padding=1,
                 dilation=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels,
                                        out_channels,
                                        kernel_size,
                                        order,
                                        padding=padding,
                                        dilation=dilation):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g.
    BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in
    order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the
    same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in
        the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        (first/second)_dilation (int or tuple): Dilation factor to create
        dilated conv, and increase the recceptive field
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 encoder,
                 kernel_size=3,
                 order='cr',
                 padding=1,
                 first_dilation=1,
                 second_dilation=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in
            # the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module(
            'SingleConv1',
            SingleConv(conv1_in_channels,
                       conv1_out_channels,
                       kernel_size,
                       order,
                       padding=padding,
                       dilation=first_dilation))
        # conv2
        self.add_module(
            'SingleConv2',
            SingleConv(conv2_in_channels,
                       conv2_out_channels,
                       kernel_size,
                       order,
                       padding=padding,
                       dilation=second_dilation))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels
    and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder
    module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 order='cr',
                 **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            order=order,
        )
        # residual block
        self.conv2 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            order=order,
        )
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'relg':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            order=n_order,
        )

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        elif 'g' in order:
            self.non_linearity = nn.GELU()
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed
    by a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_kernel_size=3,
                 apply_pooling=True,
                 pool_kernel_size=2,
                 pool_type='max',
                 basic_module=DoubleConv,
                 conv_layer_order='gcr',
                 padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg', 'strided_conv']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            elif pool_type == 'strided_conv':
                self.pooling = nn.Conv3d(in_channels,
                                         in_channels,
                                         kernel_size=1,
                                         stride=2,
                                         bias=False)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels,
                                         out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation)
    followed by a basic module (DoubleConv or ExtResNetBlock).
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must
            reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_kernel_size=3,
                 scale_factor=(2, 2, 2),
                 basic_module=DoubleConv,
                 conv_layer_order='gcr',
                 mode='nearest',
                 padding=1,
                 use_attention=False,
                 normalization="s",
                 using_bn=False):
        super(Decoder, self).__init__()
        self.use_attention = use_attention

        if basic_module == DoubleConv:
            # if DoubleConv is the basic_module use interpolation for
            # upsampling and concatenation joining
            self.upsampling = Upsampling(transposed_conv=False,
                                         in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=conv_kernel_size,
                                         scale_factor=scale_factor,
                                         mode=mode)
            # concat joining
            self.joining = partial(self._joining, concat=True)
        else:
            # if basic_module=ExtResNetBlock use transposed convolution
            # upsampling and summation joining
            self.upsampling = Upsampling(transposed_conv=True,
                                         in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=conv_kernel_size,
                                         scale_factor=scale_factor,
                                         mode=mode)
            # sum joining
            self.joining = partial(self._joining, concat=False)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels

        self.basic_module = basic_module(in_channels,
                                         out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         padding=padding)

        if use_attention:
            self.attention = AttentionConvBlock(F_g=out_channels,
                                                F_l=out_channels,
                                                F_int=out_channels // 2,
                                                F_out=1,
                                                normalization=normalization,
                                                using_bn=using_bn)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)

        if self.use_attention:
            encoder_features = self.attention(g=x, x=encoder_features)

        x = self.joining(encoder_features, x)

        x = self.basic_module(x)

        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class Upsampling(nn.Module):
    """
    Upsamples a given multi-channel 3D data using either interpolation or
    learned transposed convolution.

    Args:
        transposed_conv (bool): if True uses ConvTranspose3d for upsampling,
        otherwise uses interpolation
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'.
            Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self,
                 transposed_conv,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=3,
                 scale_factor=(2, 2, 2),
                 mode='nearest'):
        super(Upsampling, self).__init__()

        if transposed_conv:
            # make sure that the output size reverses the MaxPool3d from the
            # corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1)
        else:
            self.upsample = partial(self._interpolate, mode=mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class AttentionConvBlock(nn.Module):
    """
    3D Conv Attention Block w/ optional Normalization.
    For normalization, it supports:
    - `b` for `BatchNorm3d`,
    - `s` for `SwitchNorm3d`.
    
    `using_bn` controls SwitchNorm's behavior. It has no effect is
    `normalization == "b"`.

    SwitchNorm3d comes from:
    <https://github.com/switchablenorms/Switchable-Normalization>
    """

    def __init__(self,
                 F_g,
                 F_l,
                 F_int,
                 F_out=1,
                 normalization=None,
                 using_bn=False):
        super(AttentionConvBlock, self).__init__()

        W_g = [
            nn.Conv3d(
                F_g,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        W_x = [
            nn.Conv3d(
                F_l,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        psi = [
            nn.Conv3d(
                F_int,
                F_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        if normalization == "b":
            W_g.append(nn.BatchNorm3d(F_int))
            W_x.append(nn.BatchNorm3d(F_int))
            psi.append(nn.BatchNorm3d(F_out))
        elif normalization == "s":
            W_g.append(SwitchNorm3d(F_int, using_bn=using_bn))
            W_x.append(SwitchNorm3d(F_int, using_bn=using_bn))
            psi.append(SwitchNorm3d(F_out, using_bn=using_bn))

        self.W_g = nn.Sequential(*W_g)
        self.W_x = nn.Sequential(*W_x)

        psi.append(nn.Sigmoid())
        self.psi = nn.Sequential(*psi)

        self.gelu = nn.GELU()

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.gelu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

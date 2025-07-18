from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class UnetBasicBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )

        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out



class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        
        # ( a purple arrow in the paper )
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        
        # A light blue conv blocks in the decoder of nnUNet
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out



class UnetOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
        )

    def forward(self, inp):
        return self.conv(inp)
    

def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


class DynUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        deep_supervision: bool,
        KD: bool = False
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision
        self.KD_enabled = KD
        
        self.input_conv = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=self.in_channels,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1
                                     )
        self.down1 = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=64,
                                     out_channels=96,
                                     kernel_size=3,
                                     stride=2 # Reduces spatial dims by 2
                                     )
        self.down2 = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=96,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2
                                     )
        self.down3 = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=128,
                                     out_channels=192,
                                     kernel_size=3,
                                     stride=2
                                     )
        self.down4 = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=192,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=2
                                     )
        self.down5 = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=256,
                                     out_channels=384,
                                     kernel_size=3,
                                     stride=2
                                     )
        self.bottleneck = UnetBasicBlock( spatial_dims=self.spatial_dims,
                                     in_channels=384,
                                     out_channels=512,
                                     kernel_size=3,
                                     stride=2
                                     )
        self.up1 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=512,
                                out_channels=384,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )
        self.up2 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=384,
                                out_channels=256,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )
        self.up3 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=256,
                                out_channels=192,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )
        self.up4 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=192,
                                out_channels=128,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )
        
        self.up5 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=128,
                                out_channels=96,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )        
        self.up6 = UnetUpBlock( spatial_dims=self.spatial_dims,
                                in_channels=96,
                                out_channels=64,
                                kernel_size=3,
                                upsample_kernel_size=2
                                )
        self.out1 = UnetOutBlock( spatial_dims=self.spatial_dims,
                                  in_channels=64,
                                  out_channels=self.out_channels,
                                  )
        self.out2 = UnetOutBlock( spatial_dims=self.spatial_dims,
                                  in_channels=96,
                                  out_channels=self.out_channels,
                                  )
        self.out3 = UnetOutBlock( spatial_dims=self.spatial_dims,
                                  in_channels=128,
                                  out_channels=self.out_channels,
                                  )
        
    def forward( self, input ):
        
        # Input
        x0 = self.input_conv( input ) # x0.shape = (B x 64 x 128 x 128 x 128) or (B x 64 x 192 x 192 x 128)
        
        # Encoder
        x1 = self.down1( x0 ) # x1.shape = (B x 96 x 64 x 64 x 64)  or (B x 96 x 96 x 96 x 64)
        x2 = self.down2( x1 ) # x2.shape = (B x 128 x 32 x 32 x 32) or (B x 128 x 48 x 48 x 32)
        x3 = self.down3( x2 ) # x3.shape = (B x 192 x 16 x 16 x 16) or (B x 192 x 24 x 24 x 16)
        x4 = self.down4( x3 ) # x4.shape = (B x 256 x 8 x 8 x 8)    or (B x 256 x 12 x 12 x 8)
        x5 = self.down5( x4 ) # x5.shape = (B x 384 x 4 x 4 x 4)    or (B x 384 x 6 x 6 x 4)
        
        # Bottle-neck
        x6 = self.bottleneck( x5 ) # x6.shape = (B x 512 x 2 x 2 x 2) or (B x 512 x 3 x 3 x 2)
        
        # Decoder
        x7  = self.up1( x6, x5 )  # x7.shape  = (B x 384 x 4 x 4 x 4) or (B x 64 x 192 x 192 x 128)
        x8  = self.up2( x7, x4 )  # x8.shape  = (B x 256 x 8 x 8 x 8) or (B x 64 x 192 x 192 x 128)
        x9  = self.up3( x8, x3 )  # x9.shape  = (B x 192 x 16 x 16 x 16) or (B x 64 x 192 x 192 x 128)
        x10 = self.up4( x9, x2 )  # x10.shape = (B x 128 x 32 x 32 x 32) or (B x 64 x 192 x 192 x 128)
        x11 = self.up5( x10, x1 ) # x11.shape = (B x 96 x 64 x 64 x 64) or (B x 64 x 192 x 192 x 128)
        x12 = self.up6( x11, x0 ) # x12.shape = (B x 64 x 128 x 128 x 128) or (B x 64 x 192 x 192 x 128)
        
        # Output
        output = self.out1( x12 )
        return { 'pred' : output }
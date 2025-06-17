import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple, Union
from models.model_utils import get_conv_layer


class UnetBasicBlock(nn.Module):
    """
    A CNN module block consisting of two convolutional layers with activation and normalization,
    used as a fundamental building block in DynUNet.

    Based on:
    - `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    - `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions (2 for 2D, 3 for 3D).
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride for the first convolution. The second convolution
                always uses a stride of 1.
        norm_name: feature normalization type and arguments (default: ("INSTANCE", {"affine": True})).
        act_name: activation layer type and arguments (default: ("leakyrelu", {"inplace": True, "negative_slope": 0.01})).
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
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.lrelu = get_act_layer(name=act_name)

        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            conv_only=True
        )
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UnetBasicBlock.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying convolutions, normalization, and activation.
        """
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module typically used in the decoder path of a U-Net,
    consisting of a transposed convolution followed by a basic convolution block
    after concatenating with skip connections.

    Based on:
    - `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    - `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels (from the lower resolution path).
        out_channels: number of output channels (for the higher resolution path).
        kernel_size: convolution kernel size for the `UnetBasicBlock`.
        upsample_kernel_size: convolution kernel size for the transposed convolution layer.
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

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UnetUpBlock.

        Args:
            inp (torch.Tensor): Input tensor from the lower resolution path (e.g., bottleneck).
            skip (torch.Tensor): Skip connection tensor from the encoder path.

        Returns:
            torch.Tensor: Output tensor after upsampling, concatenation, and convolution.
        """
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(nn.Module):
    """
    Output block for the U-Net, typically a 1x1 convolution to map to the desired
    number of output classes.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels (from the last decoder block).
        out_channels: number of output channels (number of segmentation classes).
        dropout: dropout probability.
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UnetOutBlock.

        Args:
            inp (torch.Tensor): Input tensor from the last decoder layer.

        Returns:
            torch.Tensor: Output tensor representing the segmentation logits.
        """
        return self.conv(inp) 
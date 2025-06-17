import numpy as np
import torch
from typing import Optional, Sequence, Tuple, Union
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    """
    Compute padding for a convolution operation to ensure output size matches input size
    when stride is 1 and padding='same'. For general cases, it calculates padding
    based on kernel size and stride.

    Args:
        kernel_size: Kernel size of the convolution.
        stride: Stride of the convolution.

    Returns:
        Padding value or tuple of padding values.

    Raises:
        AssertionError: If calculated padding is negative.
    """
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
    """
    Compute output padding for a transposed convolution operation.

    Args:
        kernel_size: Kernel size of the transposed convolution.
        stride: Stride of the transposed convolution.
        padding: Padding applied to the transposed convolution.

    Returns:
        Output padding value or tuple of output padding values.

    Raises:
        AssertionError: If calculated output padding is negative.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


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
    """
    Returns a convolution layer (Conv or ConvTranspose) with optional activation, normalization, and dropout.

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        act: Activation layer type and arguments (e.g., Act.PRELU, ("leakyrelu", {"negative_slope": 0.01})).
        norm: Normalization layer type and arguments (e.g., Norm.INSTANCE, ("batch", {"affine": True})).
        dropout: Dropout probability or type and arguments.
        bias: Whether to use bias in the convolution.
        conv_only: If True, only returns the convolution layer without activation/norm/dropout.
        is_transposed: If True, returns a transposed convolution (ConvTranspose).

    Returns:
        monai.networks.blocks.convolutions.Convolution: The configured convolution layer.
    """
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


def set_requires_grad(nets: Union[torch.nn.Module, Sequence[torch.nn.Module]], requires_grad: bool = False):
    """
    Sets the `requires_grad` attribute for all parameters in a given network or list of networks.

    Args:
        nets (Union[torch.nn.Module, Sequence[torch.nn.Module]]): A single PyTorch module or a list of modules.
        requires_grad (bool): Whether to enable (True) or disable (False) gradient calculation for parameters.
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad 
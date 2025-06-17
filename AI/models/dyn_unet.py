import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import UnetBasicBlock, UnetUpBlock, UnetOutBlock

class DynUNet(nn.Module):
    """
    Dynamic U-Net architecture, designed for 3D medical image segmentation.
    It supports deep supervision and an optional Knowledge Distillation (KD) mode
    to extract bottleneck features.

    Args:
        spatial_dims (int): Number of spatial dimensions (e.g., 3 for 3D images).
        in_channels (int): Number of input channels (e.g., 4 for multi-modal MRI).
        out_channels (int): Number of output classes for segmentation.
        deep_supervision (bool): If True, enables deep supervision during training,
                                 outputting predictions at multiple decoder stages.
        KD (bool): If True, the model will also return the bottleneck feature map,
                   useful for Knowledge Distillation.
    """
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

        self.input_conv = UnetBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1
        )
        self.down1 = UnetBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=64,
            out_channels=96,
            kernel_size=3,
            stride=2
        )
        self.down2 = UnetBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=96,
            out_channels=128,
            kernel_size=3,
            stride=2
        )
        self.down3 = UnetBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=128,
            out_channels=192,
            kernel_size=3,
            stride=2
        )
        self.down4 = UnetBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=192,
            out_channels=256,
            kernel_size=3,
            stride=2
        )
        self.down5 = UnetBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            stride=2
        )

        self.bottleneck = UnetBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=384,
            out_channels=512,
            kernel_size=3,
            stride=2
        )

        self.up1 = UnetUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=512,
            out_channels=384,
            kernel_size=3,
            upsample_kernel_size=2
        )
        self.up2 = UnetUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            upsample_kernel_size=2
        )
        self.up3 = UnetUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=256,
            out_channels=192,
            kernel_size=3,
            upsample_kernel_size=2
        )
        self.up4 = UnetUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=192,
            out_channels=128,
            kernel_size=3,
            upsample_kernel_size=2
        )

        self.up5 = UnetUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=128,
            out_channels=96,
            kernel_size=3,
            upsample_kernel_size=2
        )
        self.up6 = UnetUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=96,
            out_channels=64,
            kernel_size=3,
            upsample_kernel_size=2
        )

        self.out1 = UnetOutBlock(
            spatial_dims=self.spatial_dims,
            in_channels=64,
            out_channels=self.out_channels,
        )
        self.out2 = UnetOutBlock(
            spatial_dims=self.spatial_dims,
            in_channels=96,
            out_channels=self.out_channels,
        )
        self.out3 = UnetOutBlock(
            spatial_dims=self.spatial_dims,
            in_channels=128,
            out_channels=self.out_channels,
        )

    def forward(self, input: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass for the DynUNet.

        Args:
            input (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            dict: A dictionary containing:
                  - 'pred': The main segmentation prediction (and optionally deep supervision predictions).
                            If deep supervision is enabled and training/KD, shape is (B, num_outputs, C_out, D, H, W).
                            Otherwise, (B, C_out, D, H, W).
                  - 'bottleneck_feature_map': (Optional) The feature map from the bottleneck layer,
                                              returned if KD is enabled.
        """
        # Input
        x0 = self.input_conv(input) # x0.shape = (B x 64 x 128 x 128 x 128)

        # Encoder
        x1 = self.down1(x0) # x1.shape = (B x 96 x 64 x 64 x 64)
        x2 = self.down2(x1) # x2.shape = (B x 128 x 32 x 32 x 32)
        x3 = self.down3(x2) # x3.shape = (B x 192 x 16 x 16 x 16)
        x4 = self.down4(x3) # x4.shape = (B x 256 x 8 x 8 x 8)
        x5 = self.down5(x4) # x5.shape = (B x 384 x 4 x 4 x 4)

        # Bottleneck
        x6 = self.bottleneck(x5) # x6.shape = (B x 512 x 2 x 2 x 2)

        # Decoder
        x7 = self.up1(x6, x5)   # x7.shape  = (B x 384 x 4 x 4 x 4)
        x8 = self.up2(x7, x4)   # x8.shape  = (B x 256 x 8 x 8 x 8)
        x9 = self.up3(x8, x3)   # x9.shape  = (B x 192 x 16 x 16 x 16)
        x10 = self.up4(x9, x2)  # x10.shape = (B x 128 x 32 x 32 x 32)
        x11 = self.up5(x10, x1) # x11.shape = (B x 96 x 64 x 64 x 64)
        x12 = self.up6(x11, x0) # x12.shape = (B x 64 x 128 x 128 x 128)

        output1 = self.out1(x12)

        return_dict = {'pred': output1}

        if (self.training and self.deep_supervision) or self.KD_enabled:
            output2 = F.interpolate(self.out2(x11), size=output1.shape[2:])
            output3 = F.interpolate(self.out3(x10), size=output1.shape[2:])
            output_all = [output1, output2, output3]
            return_dict['pred'] = torch.stack(output_all, dim=1)

        if self.KD_enabled:
            return_dict['bottleneck_feature_map'] = x6

        return return_dict 
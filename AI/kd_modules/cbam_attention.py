import torch
import torch.nn as nn

class CBAMFeatureExtractor(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) for 3D feature maps.
    CBAM sequentially applies Channel Attention Module (CAM) and Spatial Attention Module (SAM)
    to input features, allowing the network to emphasize meaningful features along
    both channel and spatial axes.

    Reference: "CBAM: Convolutional Block Attention Module" by Woo et al. (2018)

    Args:
        in_channels (int): Number of input channels for the feature map.
        reduction (int): Reduction ratio for the Channel Attention Module's MLP.
                         (Default: 16)
        kernel_size (int): Kernel size for the Spatial Attention Module's convolution.
                           (Default: 7)
    """
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAMFeatureExtractor, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CBAMFeatureExtractor.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Attention-enhanced feature map of the same shape as input.
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attention = avg_out + max_out
        x = x * channel_attention

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attention

        return x
    
# class SEFeatureExtractor(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEFeatureExtractor, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         batch_size, C, H, W, D = x.shape
#         y = self.avg_pool(x).view(batch_size, C)
#         y = self.fc(y).view(batch_size, C, 1, 1, 1)
#         return x * y  # Scale input features

# class TransformerFeatureExtractor(nn.Module):
#     def __init__(self, in_channels, num_heads=4):
#         super(TransformerFeatureExtractor, self).__init__()
#         self.norm = nn.LayerNorm(in_channels)
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)

#     def forward(self, x):
#         batch_size, C, H, W, D = x.shape
#         x = x.view(batch_size, C, -1).permute(0, 2, 1)  # Flatten spatial dims & permute for attention
#         x = self.norm(x)
#         attn_output, _ = self.multihead_attn(x, x, x)
#         return attn_output.permute(0, 2, 1).view(batch_size, C, H, W, D)

# class NonLocalFeatureExtractor(nn.Module):
#     def __init__(self, in_channels):
#         super(NonLocalFeatureExtractor, self).__init__()
#         self.inter_channels = in_channels // 2  # Reduce feature dimensions

#         self.theta = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1, bias=False)
#         self.phi = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1, bias=False)
#         self.g = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1, bias=False)
#         self.out_conv = nn.Conv3d(self.inter_channels, in_channels, kernel_size=1, bias=False)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         batch_size, C, H, W, D = x.shape
#         spatial_dims = H * W * D
#         theta_x = self.theta(x).view(batch_size, self.inter_channels, spatial_dims)  # (B, C/2, N)
#         phi_x = self.phi(x).view(batch_size, self.inter_channels, spatial_dims)  # (B, C/2, N)
#         theta_x = theta_x.permute(0, 2, 1)  # (B, N, C/2)
#         attention = self.softmax(torch.bmm(theta_x, phi_x))  # (B, N, N)
#         g_x = self.g(x).view(batch_size, self.inter_channels, spatial_dims).permute(0, 2, 1)  # (B, N, C/2)
#         out = torch.bmm(attention, g_x)  # (B, N, C/2)
        
#         # Reshape back to (B, C/2, H, W, D)
#         out = out.permute(0, 2, 1).view(batch_size, self.inter_channels, H, W, D)  # (B, C/2, H, W, D)
#         out = self.out_conv(out)  # (B, C, H, W, D)

#         return x + out  # Residual connection

# class GatedAttentionFeatureExtractor(nn.Module):
#     def __init__(self, in_channels):
#         super(GatedAttentionFeatureExtractor, self).__init__()
#         self.attention_gate = nn.Conv3d(in_channels, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         gate = self.sigmoid(self.attention_gate(x))
#         return x * gate  # Scale input by attention

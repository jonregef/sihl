import torch
from torch import nn, Tensor


class SpatialAttention(nn.Module):
    """https://arxiv.org/abs/1807.06521"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True).values
        return self.conv(torch.cat([avg_out, max_out], dim=1)).sigmoid()


class ChannelAttention(nn.Module):
    """https://arxiv.org/abs/1807.06521"""

    def __init__(self, in_channels: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))
        return (avg_out + max_out).sigmoid()


class CBAM(nn.Module):
    """https://arxiv.org/abs/1807.06521"""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 7,
        ratio: int = 16,
        applied: bool = False,
    ) -> None:
        super().__init__()
        self.applied = applied
        self.channel_attention = ChannelAttention(in_channels, kernel_size)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attention(x) * x
        attention = self.spatial_attention(x) * x
        return attention * x if self.applied else attention


class CrossCBAM(nn.Module):
    """https://arxiv.org/abs/2306.02306"""

    def __init__(self, in_channels: int, kernel_size: int = 7, ratio: int = 16) -> None:
        super().__init__()
        self.channel_attention_high = ChannelAttention(in_channels, ratio)
        self.spatial_attention_high = SpatialAttention(kernel_size)
        self.channel_attention_low = ChannelAttention(in_channels, ratio)
        self.spatial_attention_low = SpatialAttention(kernel_size)

    def forward(self, low: Tensor, high: Tensor) -> Tensor:
        mid_low = self.channel_attention_low(low) * high
        mid_high = self.channel_attention_high(high) * low
        out_low = self.spatial_attention_low(mid_low) * mid_high
        out_high = self.spatial_attention_high(mid_high) * mid_low
        return out_low + out_high

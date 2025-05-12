from functools import partial
from typing import List

from einops import rearrange
from torch import Tensor
from torch.nn import functional
from torchvision import ops
import torch
import torch.nn as nn

from sihl.utils import sine_embedding, coordinate_grid


class HybridEncoder(nn.Module):
    """https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/src/zoo/rtdetr/hybrid_encoder.py"""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        bottom_level: int,
        top_level: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.top_in_level = min(top_level, len(in_channels) - 1)

        self.bottom_level, self.top_level = bottom_level, top_level
        levels = range(bottom_level, top_level + 1)
        self.num_channels = out_channels
        self.out_channels = in_channels.copy()
        self.out_channels[levels.start : levels.stop] = [out_channels for _ in levels]

        Conv = partial(ops.Conv2dNormActivation, activation_layer=nn.SiLU)
        self.input_projections = nn.ModuleList(
            Conv(in_channels[level], out_channels, 1)
            for level in range(bottom_level, self.top_in_level + 1)
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                out_channels,
                nhead=4,
                dim_feedforward=1024,
                dropout=0,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=1,
        )

        # top-down (fpn)
        self.up_convs = nn.ModuleList()
        self.up_fusions = nn.ModuleList()
        for _ in range(self.top_in_level, bottom_level, -1):
            self.up_convs.append(Conv(out_channels, out_channels, 1))
            self.up_fusions.append(CSPRepLayer(out_channels * 2, out_channels))

        self.extra_downscalers = nn.ModuleList(
            Conv(out_channels, out_channels, 3, stride=2)
            for _ in range(top_level - len(in_channels) + 1)
        )

        # bottom-up (pan)
        self.down_convs = nn.ModuleList()
        self.down_fusions = nn.ModuleList()
        for _ in range(bottom_level, top_level):
            self.down_convs.append(Conv(out_channels, out_channels, 3, stride=2))
            self.down_fusions.append(CSPRepLayer(out_channels * 2, out_channels))

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        xs = inputs[self.bottom_level : self.top_in_level + 1]
        xs = [project(x) for project, x in zip(self.input_projections, xs)]

        batch_size, _, height, width = xs[-1].shape
        x = rearrange(xs[-1], "b c h w -> b (h w) c")
        grid = rearrange(coordinate_grid(height, width).to(x), "h w c -> (h w) c")
        pos_emb = sine_embedding(grid, self.num_channels).detach()
        x = self.encoder(x + pos_emb)
        x = rearrange(x, "b (h w) c -> b c h w", h=height, w=width)
        xs = xs[:-1] + [x]

        inner_outs = [x]
        for idx, (conv, fuse) in enumerate(zip(self.up_convs, self.up_fusions)):
            feat_low = xs[len(xs) - 2 - idx]
            feat_high = conv(inner_outs[0])
            inner_outs[0] = feat_high
            feat_high = functional.interpolate(feat_high, scale_factor=2)
            inner_out = fuse(feat_high, feat_low)
            inner_outs.insert(0, inner_out)

        for downscaler in self.extra_downscalers:
            inner_outs.append(downscaler(inner_outs[-1]))

        outs = [inner_outs[0]]
        for idx, (conv, fuse) in enumerate(zip(self.down_convs, self.down_fusions)):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = conv(feat_low)
            out = fuse(downsample_feat, feat_high)
            outs.append(out)

        return inputs[: self.bottom_level] + outs + inputs[self.top_level + 1 :]


class RepVggBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        Conv = partial(ops.Conv2dNormActivation, activation_layer=None)
        self.conv1 = Conv(in_channels, out_channels, 3)
        self.conv2 = Conv(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return functional.silu(self.conv1(x) + self.conv2(x))


class CSPRepLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3):
        super().__init__()
        Conv = partial(ops.Conv2dNormActivation, activation_layer=nn.SiLU)
        self.conv1 = Conv(in_channels, out_channels, 1)
        self.conv2 = Conv(in_channels, out_channels, 1)
        self.bottlenecks = nn.Sequential(
            *[RepVggBlock(out_channels, out_channels) for _ in range(num_layers)]
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.cat([x1, x2], dim=1)
        return self.bottlenecks(self.conv1(x)) + self.conv2(x)

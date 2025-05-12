from typing import List
from functools import partial

from torch import nn, Tensor

from sihl.layers.convblocks import ConvNormAct
from sihl.layers.scalers import Interpolate


class FPN(nn.Module):
    """https://arxiv.org/abs/1612.03144"""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        bottom_level: int,
        top_level: int,
        norm: str = "batch",
        act: str = "silu",
    ):
        super().__init__()
        assert 0 < bottom_level < top_level
        self.in_levels = range(bottom_level, min(top_level + 1, len(in_channels)))
        self.bottom_level, self.top_level = bottom_level, top_level
        levels = range(bottom_level, top_level + 1)
        self.out_channels = in_channels.copy()
        self.out_channels[levels.start : levels.stop] = [out_channels for _ in levels]

        Conv = partial(ConvNormAct, norm=norm, act=act)
        self.input_projections = nn.ModuleList(
            Conv(in_channels[level], out_channels, 1) for level in self.in_levels
        )
        self.up_convs = nn.ModuleList(
            Conv(out_channels, out_channels, 1) for level in self.in_levels[:-1]
        )
        self.upscalers = nn.ModuleList(
            Interpolate(scale=2) for level in self.in_levels[:-1]
        )
        self.extra_downscalers = nn.ModuleList(
            Conv(out_channels, out_channels, stride=2)
            for _ in range(top_level - len(in_channels) + 1)
        )
        self.out_convs = nn.ModuleList(Conv(out_channels, out_channels) for _ in levels)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        xs = inputs[self.in_levels.start : self.in_levels.stop]
        xs = [project(x) for project, x in zip(self.input_projections, xs)]

        top_down = [xs[-1]]
        for i, (conv, up) in enumerate(zip(self.up_convs, self.upscalers)):
            top_down[i] = conv(top_down[i])
            top_down.append(up(top_down[i]) + xs[-(i + 2)])

        top_down = top_down[::-1]
        for down in self.extra_downscalers:
            top_down.append(down(top_down[-1]))

        top_down = [conv(feat) for conv, feat in zip(self.out_convs, top_down)]
        return inputs[: self.bottom_level] + top_down + inputs[self.top_level + 1 :]

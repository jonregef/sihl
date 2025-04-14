from typing import List

from torch import nn, Tensor

from sihl.layers.convblocks import ConvNormAct
from sihl.layers.fpn import FPN


class PAN(FPN):
    """https://arxiv.org/abs/1803.01534"""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        bottom_level: int,
        top_level: int,
        norm: str = "batch",
        act: str = "silu",
    ):
        super().__init__(in_channels, out_channels, bottom_level, top_level, norm, act)
        self.downscalers = nn.ModuleList(
            ConvNormAct(out_channels, out_channels, stride=2, norm=norm, act=act)
            for level in range(bottom_level, top_level)
        )

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        inputs = super().forward(inputs)
        xs = inputs[self.bottom_level : self.top_level + 1]
        outputs = [xs[0]]
        for i, downscale in enumerate(self.downscalers):
            outputs.append(downscale(outputs[-1]) + xs[i + 1])
        return inputs[: self.bottom_level] + outputs + inputs[self.top_level + 1 :]

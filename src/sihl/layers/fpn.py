from typing import List

from torch import nn, Tensor

from sihl.layers.convblocks import ConvNormAct
from sihl.layers.scalers import BilinearScaler, AntialiasedDownscaler


class FPN(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        bottom_level: int,
        top_level: int,
    ):
        super().__init__()
        assert 0 < bottom_level < top_level
        self.out_channels = (
            in_channels[:bottom_level]
            + [out_channels for _ in range(bottom_level, top_level + 1)]
            + in_channels[top_level:]
        )
        self.bottom_level, self.top_level = bottom_level, top_level
        self.lateral_convs = nn.ModuleList(
            ConvNormAct(in_c, out_channels, kernel_size=1)
            for in_c in in_channels[bottom_level : top_level + 1]
        )
        self.upscalers = nn.ModuleList(
            BilinearScaler(scale=2) for level in range(len(self.lateral_convs) - 1)
        )
        self.downscalers = nn.ModuleList(
            AntialiasedDownscaler(out_channels, out_channels)
            for _ in range(top_level + 1 - len(in_channels))
        )
        self.convs = nn.ModuleList(
            ConvNormAct(out_channels, out_channels)
            for _ in range(top_level - bottom_level + 1)
        )

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        features = [
            lateral_conv(inputs[self.bottom_level + idx])
            for idx, lateral_conv in enumerate(self.lateral_convs)
        ]
        for idx, upscaler in enumerate(self.upscalers):
            features[-idx - 2] = upscaler(features[-idx - 1]) + features[-idx - 2]
        for downscaler in self.downscalers:
            features.append(downscaler(features[-1]))
        outputs = [conv(x) for conv, x in zip(self.convs, features)]
        return inputs[: self.bottom_level] + outputs + inputs[self.top_level + 1 :]

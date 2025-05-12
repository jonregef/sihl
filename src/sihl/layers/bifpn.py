from typing import List

from torch import nn, Tensor
import torch

from sihl.layers.convblocks import ConvNormAct
from sihl.layers.scalers import Interpolate, AntialiasedDownscaler


class FastNormalizedFusion(nn.Module):
    def __init__(self, num_inputs: int = 2) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs), requires_grad=True)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        weights = self.weights.to(inputs[0].device).softmax(dim=0)
        return torch.stack([w * x for w, x in zip(weights, inputs)]).sum(dim=0)


class BiFPNLayer(nn.Module):
    def __init__(self, out_channels: int, num_levels: int, **kwargs) -> None:
        super().__init__()
        assert num_levels > 1, num_levels
        self.num_levels = num_levels
        levels = range(num_levels - 1)
        self.upscalers = nn.ModuleList(Interpolate(scale=2) for _ in levels)
        self.up_fusions = nn.ModuleList(FastNormalizedFusion(2) for _ in levels)
        self.up_convs = nn.ModuleList(
            ConvNormAct(out_channels, out_channels, **kwargs) for _ in levels
        )
        self.downscalers = nn.ModuleList(
            AntialiasedDownscaler(out_channels, out_channels, **kwargs) for _ in levels
        )
        self.down_fusions = nn.ModuleList(FastNormalizedFusion(3) for _ in levels)
        self.down_convs = nn.ModuleList(
            [ConvNormAct(out_channels, out_channels, **kwargs) for _ in levels]
        )

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        assert len(inputs) == self.num_levels
        top_down = [inputs[-1]]
        for idx, (conv, fuse, upscale) in enumerate(
            zip(self.up_convs, self.up_fusions, self.upscalers)
        ):
            top_down.append(conv(fuse([upscale(top_down[-1]), inputs[-2 - idx]])))
        top_down = top_down[::-1]  # re-order to put lowest level first
        bottom_up = [top_down[0]]
        for idx, (conv, fuse, downscale) in enumerate(
            zip(self.down_convs, self.down_fusions, self.downscalers)
        ):
            args = [downscale(bottom_up[-1]), inputs[idx + 1], top_down[idx + 1]]
            bottom_up.append(conv(fuse(args)))
        return bottom_up


class BiFPN(nn.Module):
    """https://arxiv.org/abs/1911.09070"""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        bottom_level: int,
        top_level: int,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        assert num_layers > 0
        assert 0 < bottom_level < top_level
        self.out_channels = in_channels[:bottom_level] + [
            out_channels for level in range(bottom_level, top_level + 1)
        ]
        self.bottom_level = bottom_level
        self.top_level = top_level
        self.lateral_connections = nn.ModuleList(
            ConvNormAct(in_c, out_channels, kernel_size=1, **kwargs)
            for in_c in in_channels[bottom_level : top_level + 1]
        )
        self.downscalers = nn.ModuleList(
            AntialiasedDownscaler(out_channels, out_channels, **kwargs)
            for _ in range(top_level + 1 - len(in_channels))
        )
        num_levels = top_level - bottom_level + 1
        self.layers = nn.Sequential(
            *(BiFPNLayer(out_channels, num_levels, **kwargs) for _ in range(num_layers))
        )

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        features = [
            lateral(inputs[self.bottom_level + idx])
            for idx, lateral in enumerate(self.lateral_connections)
        ]
        for downscaler in self.downscalers:
            features.append(downscaler(features[-1]))
        outs = self.layers(features)
        return inputs[: self.bottom_level] + outs + inputs[self.top_level + 1 :]

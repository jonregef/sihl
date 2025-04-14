from typing import Union, Optional, Tuple, Literal

from einops import reduce
from torch import Tensor, nn
from torch.nn.functional import interpolate

from sihl.layers.convblocks import ConvNormAct
from sihl.layers.pooling import BlurPool2d


class StridedDownscaler(ConvNormAct):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__(in_channels, out_channels, stride=2, **kwargs)


class AntialiasedDownscaler(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, **kwargs
    ) -> None:
        super().__init__(
            ConvNormAct(in_channels, out_channels, kernel_size, **kwargs),
            BlurPool2d(out_channels, stride=2),
        )


class SimpleDownscaler(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, **kwargs
    ) -> None:
        super().__init__(
            ConvNormAct(in_channels, out_channels, kernel_size, **kwargs),
            nn.AvgPool2d(2, stride=2),
        )


class Interpolate(nn.Module):
    def __init__(
        self,
        scale: Optional[Union[float, int]] = None,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        mode: Literal["nearest", "bilinear"] = "bilinear",
    ) -> None:
        super().__init__()
        self.scale, self.size, self.mode = scale, size, mode

    def forward(self, x: Tensor) -> Tensor:
        return interpolate(x, scale_factor=self.scale, size=self.size, mode=self.mode)


class SimpleUpscaler(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__(
            Interpolate(scale=2), ConvNormAct(in_channels, out_channels, kernel_size)
        )


class BilinearAdditiveUpscaler(nn.Module):
    """[The Devil is in the Decoder](https://arxiv.org/abs/1707.05847)"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.upscaler = Interpolate(scale=2)
        self.residual = nn.ConvTranspose2d(
            in_channels, in_channels // 4, kernel_size=2, stride=2
        )
        self.out_conv = ConvNormAct(
            in_channels // 4, out_channels, kernel_size=kernel_size
        )

    def forward(self, x: Tensor) -> Tensor:
        a = reduce(self.upscaler(x), "b (c1 c) h w -> b c h w", "mean", c1=4)
        b = self.residual(x)
        return self.out_conv(a + b)

from typing import Type, Protocol, Literal, Optional

from torch import nn, Tensor


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, groups=groups, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding: Optional[int] = None,
        norm: Literal["batch", "group", None] = "batch",
        act: Literal["relu", "silu", "sigmoid", "softplus", "softmax", None] = "relu",
        bias: Optional[int] = None,
        conv_layer: nn.Module = nn.Conv2d,
    ) -> None:
        conv_layer = nn.Conv2d if kernel_size == 1 else conv_layer
        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding or ((kernel_size - 1) // 2 * dilation),
                dilation=dilation,
                groups=groups,
                bias=bias or (norm is None),
            )
        ]

        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == "group":
            layers.append(nn.GroupNorm(32, out_channels))

        if act == "relu":
            layers.append(nn.ReLU())
        elif act == "silu":
            layers.append(nn.SiLU())
        elif act == "sigmoid":
            layers.append(nn.Sigmoid())
        elif act == "softplus":
            layers.append(nn.Softplus())
        elif act == "softmax":
            layers.append(nn.Softmax(dim=1))

        super().__init__(*layers)


class HasInOutChannels(Protocol):
    def __init__(
        self, in_channels: int, out_channels: int, *args, **kwargs
    ) -> None: ...


class SequentialConvBlocks(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        ConvBlock: Type[HasInOutChannels] = ConvNormAct,
        **kwargs,
    ) -> None:
        if num_layers <= 0:
            super().__init__(nn.Identity())
        else:
            super().__init__(
                ConvBlock(in_channels, out_channels, kernel_size=kernel_size, **kwargs),
                *[
                    ConvBlock(
                        out_channels, out_channels, kernel_size=kernel_size, **kwargs
                    )
                    for _ in range(num_layers - 1)
                ],
            )

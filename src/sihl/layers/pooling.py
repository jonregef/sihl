from torch import nn, Tensor
from torch.nn import functional
import numpy as np
import torch


class BlurPool2d(nn.Module):
    """https://arxiv.org/abs/1904.11486"""

    def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = [conv_padding(kernel_size, stride, dilation=1)] * 4
        coeffs = torch.tensor(
            (np.poly1d((0.5, 0.5)) ** (kernel_size - 1)).coeffs.astype(np.float32)
        )
        kernel = (coeffs[:, None] * coeffs[None, :])[None, None, :, :]
        self.register_buffer("kernel", kernel.repeat(in_channels, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = functional.pad(x, self.padding, "reflect")
        return functional.conv2d(
            x, self.kernel.to(x.device), stride=self.stride, groups=self.in_channels
        )


def conv_padding(kernel_size: int, stride: int, dilation: int) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2

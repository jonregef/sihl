from typing import List

from torch import Tensor, nn
from torch.nn import functional
import torch


class Normalize(nn.Module):
    def __init__(self, mean: List[float], std: List[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).reshape(1, -1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class PadToMultipleOf(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: Tensor) -> Tensor:
        pad_x = (self.n - x.shape[3] % self.n) % self.n
        pad_y = (self.n - x.shape[2] % self.n) % self.n
        return functional.pad(
            x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2)
        )

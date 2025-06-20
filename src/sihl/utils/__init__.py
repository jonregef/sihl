# ruff: noqa: F401
from typing import Any, Tuple, List, Callable, Union, Optional, Literal
import functools
import math
import random
import sys

from einops import rearrange
from torch import nn, Tensor
from torch.nn import functional
from torch.nn.functional import avg_pool2d
import torch
import torchvision

from sihl.utils.polygon_iou import polygon_iou
from sihl.utils.pck import PercentageOfCorrectKeypoints
from sihl.utils.f1 import OptimalF1Threshold


EPS = 1e-5


class BatchedMeanVarianceAccumulator:
    """https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.M2 = None

    def update(self, x: Tensor) -> None:
        if self.mean is None:
            self.mean = x.mean(dim=0)
            self.M2 = torch.zeros_like(self.mean)
        else:
            if x.shape[1:] != self.mean.shape:
                raise ValueError(
                    f"Shape mismatch: got {x.shape[1:]}, expected {self.mean.shape}"
                )

            batch_count = x.size(0)
            total_count = self.count + batch_count
            batch_mean = x.mean(dim=0)
            batch_delta = batch_mean - self.mean
            self.mean += batch_delta * batch_count / total_count

            self.M2 += (
                x.var(dim=0, unbiased=False) * batch_count
                + (batch_delta**2) * self.count * batch_count / total_count
            )

        self.count += x.size(0)

    def compute(self):
        if self.count < 2:
            return self.mean, torch.full_like(self.mean, float("nan"))
        variance = self.M2 / (self.count - 1)
        return self.mean, variance


def random_pad(
    image: Tensor, target_size: Union[int, Tuple[int, int]], fill: Union[float, int] = 0
) -> Tensor:
    image_size = image.shape[1:]
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    if image_size[0] > target_size[0] or image_size[1] > target_size[1]:
        scale_factor = min(
            target_size[0] / image_size[0], target_size[1] / image_size[1]
        )
        new_height = int(image_size[0] * scale_factor)
        new_width = int(image_size[1] * scale_factor)
        image = torchvision.transforms.functional.resize(image, (new_height, new_width))
        image_size = image.shape[1:]

    pad_width = target_size[1] - image_size[1]
    pad_height = target_size[0] - image_size[0]

    top_pad = random.randint(0, pad_height)
    bottom_pad = pad_height - top_pad
    left_pad = random.randint(0, pad_width)
    right_pad = pad_width - left_pad

    padded_image = torchvision.transforms.functional.pad(
        image, (left_pad, top_pad, right_pad, bottom_pad), fill
    )
    return padded_image


def coordinate_grid(height: int, width: int) -> Tensor:
    """2D grid of normalized pixel center coordinates."""
    y_min, x_min = 1 / height / 2, 1 / width / 2
    ys = torch.linspace(y_min, 1 - y_min, steps=height)[:, None].repeat(1, width)
    xs = torch.linspace(x_min, 1 - x_min, steps=width)[None, :].repeat(height, 1)
    return torch.stack([xs, ys], dim=2)


def sine_embedding_1d(positions: int, dim: int, temperature=10000, device=None):
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {dim}")

    if positions.dim() == 0:
        positions = positions.unsqueeze(0)

    if device is None:
        device = positions.device

    half_dim = dim // 2
    emb = math.log(temperature) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = positions.unsqueeze(-1).to(torch.float) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


def sine_embedding_2d(
    height_pos: int, width_pos: int, dim: int, temperature=10000, device=None
):
    if dim % 4 != 0:
        raise ValueError(f"Embedding dimension must be divisible by 4, got {dim}")
    if device is None:
        device = height_pos.device
    dim_per_axis = dim // 2
    height_emb = sine_embedding_1d(height_pos, dim_per_axis, temperature, device)
    width_emb = sine_embedding_1d(width_pos, dim_per_axis, temperature, device)
    emb_2d = torch.cat([height_emb, width_emb], dim=-1)
    return emb_2d


def sine_embedding_2d_grid(
    height: int, width: int, dim: int, temperature=10000, device=None
):
    y_pos = torch.arange(height, dtype=torch.float32, device=device)
    x_pos = torch.arange(width, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")
    emb_2d = sine_embedding_2d(y_grid, x_grid, dim, temperature, device)
    return emb_2d


def f_score(beta: float) -> Callable[[Tensor, Tensor], Tensor]:
    """https://en.wikipedia.org/wiki/F-score#Definition"""
    return lambda p, r: (1 + beta**2) * p * r / (beta**2 * p + r)


def points_to_bbox(points: Tensor) -> Tensor:
    """(N, K, 2) -> (N, 4)"""
    min_x = torch.min(points[..., 0], dim=1).values
    min_y = torch.min(points[..., 1], dim=1).values
    max_x = torch.max(points[..., 0], dim=1).values
    max_y = torch.max(points[..., 1], dim=1).values
    return torch.stack([min_x, min_y, max_x, max_y], dim=1)


def edges(x: Tensor) -> Tensor:
    sobel_kernel_x = torch.tensor(
        [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=x.dtype, device=x.device
    )
    sobel_kernel_y = torch.tensor(
        [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=x.dtype, device=x.device
    )
    sobel_kernel_x = sobel_kernel_x.repeat(x.shape[1], 1, 1, 1)
    sobel_kernel_y = sobel_kernel_y.repeat(x.shape[1], 1, 1, 1)
    edges_x = functional.conv2d(x, sobel_kernel_x, padding=1, groups=x.shape[1])
    edges_y = functional.conv2d(x, sobel_kernel_y, padding=1, groups=x.shape[1])
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    edges = edges / edges.max()
    return edges


def gaussian_blur(
    x: Tensor, kernel_size: int = 5, sigma: Optional[float] = None
) -> Tensor:
    sigma = sigma or 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    kernel_1d = torch.exp(
        -(torch.arange(-(kernel_size // 2), kernel_size // 2 + 1) ** 2) / (2 * sigma**2)
    )
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.expand(x.shape[1], -1, -1, -1).to(x)
    return functional.conv2d(x, kernel_2d, padding=kernel_size // 2, groups=x.shape[1])


def ssim_loss(
    pred: Tensor, gt: Tensor, window_size: int = 11, size_average: bool = True
) -> Tensor:
    C1, C2 = 0.01**2, 0.03**2
    padding = window_size // 2
    mu1 = avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu2 = avg_pool2d(gt, window_size, stride=1, padding=padding)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu1_sq
    sigma2_sq = avg_pool2d(gt * gt, window_size, stride=1, padding=padding) - mu2_sq
    sigma12 = avg_pool2d(pred * gt, window_size, stride=1, padding=padding) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return (1 - ssim_map.mean() if size_average else 1 - ssim_map.sum()).abs()


def focal_loss(
    preds: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> Tensor:
    """https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html"""
    with torch.autocast(device_type="cuda", enabled=False):
        preds, targets = preds.to(torch.float32), targets.to(torch.float32)
        ce_loss = functional.binary_cross_entropy(preds, targets, reduction="none")
        p_t = preds * targets + (1 - preds) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * ce_loss * ((1 - p_t) ** gamma)
    return loss


def tversky_loss(
    preds: Tensor,
    targets: Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    valid_mask = targets != ignore_index
    targets = functional.one_hot(targets * valid_mask, preds.shape[1])
    targets = targets.permute(0, 3, 1, 2) * valid_mask.unsqueeze(1)
    preds = functional.softmax(preds, dim=1) * valid_mask.unsqueeze(1)
    tp = (preds * targets).sum(dim=(2, 3))
    fn = ((1 - preds) * targets).sum(dim=(2, 3))
    fp = (preds * (1 - targets)).sum(dim=(2, 3))
    return 1 - ((tp + EPS) / (tp + alpha * fp + beta * fn + EPS)).mean()


def recursive_getattr(obj: Any, attr: str, *args):
    """https://stackoverflow.com/a/31174427"""

    def _getattr(obj: Any, attr: str):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recursive_setattr(obj: Any, attr: str, val: Any):
    """https://stackoverflow.com/a/31174427"""
    pre, _, post = attr.rpartition(".")
    return setattr(recursive_getattr(obj, pre) if pre else obj, post, val)


def call_and_get_locals(func: Callable, *args, **kwargs) -> Tuple[Any, List[Any]]:
    """https://stackoverflow.com/a/52358426"""
    frame = None

    def snatch_locals(_frame, name, arg):
        nonlocal frame
        if name == "call":
            frame = _frame
            sys.settrace(sys._getframe(0).f_trace)
        return sys._getframe(0).f_trace

    sys.settrace(snatch_locals)

    try:
        result = func(*args, **kwargs)
    finally:
        sys.settrace(sys._getframe(0).f_trace)

    return result, frame.f_locals

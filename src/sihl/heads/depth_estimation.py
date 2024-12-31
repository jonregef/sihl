from typing import List, Tuple, Dict

from einops import rearrange, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
import torch

from sihl.heads.semantic_segmentation import SemanticSegmentation
from sihl.layers import SequentialConvBlocks
from sihl.utils import interpolate, EPS


class DepthEstimation(SemanticSegmentation):
    """Depth estimation is pixelwise regression.

    Refs:
        1. [Adabins](https://arxiv.org/abs/2011.14141)
    """

    def __init__(
        self,
        in_channels: List[int],
        lower_bound: float,
        upper_bound: float,
        bottom_level: int = 3,
        top_level: int = 5,
        num_channels: int = 256,
        num_layers: int = 1,
        num_bins: int = 256,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            lower_bound (float): Lower bound of the interval of possible values.
            upper_bound (float): Upper bound of the interval of possible values.
            bottom_level (int, optional): Bottom level of inputs this head is attached to. Defaults to 3.
            top_level (int, optional): Top level of inputs this head is attached to. Defaults to 7.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 4.
            num_bins (int, optional): Number of dynamic-sized bins. Defaults to 256.
        """
        assert lower_bound < upper_bound
        assert len(in_channels) > top_level >= bottom_level > 0
        assert num_channels > 0 and num_layers > 0
        assert num_bins > 1
        super().__init__(
            in_channels=in_channels,
            num_classes=num_bins,
            num_channels=num_channels,
            bottom_level=bottom_level,
            top_level=top_level,
            num_layers=num_layers,
        )

        self.num_bins = num_bins
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        self.bin_head = nn.Sequential(
            SequentialConvBlocks(in_channels[top_level], num_channels, num_layers),
            nn.Conv2d(num_channels, num_bins, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        stride = 2**bottom_level
        self.output_shapes = {
            "depth_maps": ("batch_size", f"height/{stride}", f"width/{stride}")
        }

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def denormalize(self, x: Tensor) -> Tensor:
        return x * (self.upper_bound - self.lower_bound) + self.lower_bound

    def get_bin_centers(self, inputs: List[Tensor]) -> Tensor:
        bin_widths = self.bin_head(inputs[self.top_level]).relu() + EPS
        bin_widths = bin_widths / reduce(bin_widths, "n c -> n 1", "sum")
        return bin_widths.cumsum(dim=1) - bin_widths / 2

    def get_depth_map(self, inputs: List[Tensor], bin_centers: Tensor) -> Tensor:
        weights = self.get_logits(inputs).relu() + EPS
        weights = weights / reduce(weights, "n c h w -> n 1 h w", "sum")
        bin_centers = rearrange(bin_centers, "n c -> n c 1 1")
        depth_map = reduce(bin_centers * weights, "n c h w -> n 1 h w", "sum")
        return depth_map.clamp(0, 1)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        bin_centers = self.get_bin_centers(inputs)
        depth_map = self.denormalize(self.get_depth_map(inputs, bin_centers))
        return interpolate(depth_map, size=inputs[0].shape[2:]).squeeze(1)

    def training_step(
        self, inputs: List[Tensor], targets: Tensor, masks: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        batch_size, target_height, target_width = targets.shape
        masks = rearrange(masks, "b h w -> b 1 h w")
        targets = rearrange(targets, "b h w -> b 1 h w")
        targets = self.normalize(targets)

        bin_centers = self.get_bin_centers(inputs)
        depth_map = self.get_depth_map(inputs, bin_centers=bin_centers)
        pred_shape = depth_map.shape[2:]
        depth_map = interpolate(depth_map, size=targets.shape[2:])

        g = (depth_map[masks] + EPS).log() - (targets[masks] + EPS).log()
        pix_loss = torch.sqrt(g.var() + 0.15 * g.mean().pow(2)) * 10

        masks = functional.interpolate(
            masks.to(torch.uint8), size=pred_shape, mode="nearest-exact"
        ).to(torch.bool)
        targets = interpolate(targets, size=pred_shape)
        hist_losses = []  # bidirectional chamfer loss
        for batch_idx in range(batch_size):
            target_hist = rearrange(targets[batch_idx][masks[batch_idx]], "k -> k 1")
            pred_hist = rearrange(bin_centers[batch_idx], "l -> 1 l")
            dist = (pred_hist - target_hist).pow(2)
            forward_chamfer = reduce(dist, "k l -> k", "min").mean()
            backward_chamfer = reduce(dist, "k l -> l", "min").mean()
            hist_losses.append(forward_chamfer + backward_chamfer)
        hist_loss = torch.stack(hist_losses).mean()
        return pix_loss + hist_loss, {"pixel_loss": pix_loss, "hist_loss": hist_loss}

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.mae_computer = MeanAbsoluteError()
        self.rmse_computer = MeanSquaredError(squared=False)

    def validation_step(
        self, inputs: List[Tensor], targets: Tensor, masks: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss, _ = self.training_step(inputs, targets, masks)
        device = loss.device
        self.loss_computer.to(device).update(loss)
        depth_map = self.forward(inputs)
        batch_size, pred_height, pred_width = depth_map.shape
        self.rmse_computer.to(device).update(depth_map[masks], targets[masks])
        self.mae_computer.to(device).update(depth_map[masks], targets[masks])
        return loss, {}

    def on_validation_end(self) -> Dict[str, float]:
        return {
            "loss": self.loss_computer.compute().item(),
            "rmse": self.rmse_computer.compute().item(),
            "mae": self.mae_computer.compute().item(),
        }

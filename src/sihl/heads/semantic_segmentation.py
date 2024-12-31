from typing import Tuple, List, Union, Dict

from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import JaccardIndex, MeanMetric, Accuracy
import torch

from sihl.layers import (
    ConvNormAct,
    SequentialConvBlocks,
    SimpleUpscaler,
    BilinearScaler,
)
from sihl.utils import interpolate


class SemanticSegmentation(nn.Module):
    """Semantic segmentation is pixelwise multiclass classification.

    Refs:
        1. [PP-LiteSeg](https://arxiv.org/abs/2204.02681)
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        bottom_level: int = 3,
        top_level: int = 5,
        num_channels: int = 256,
        num_layers: int = 3,
        pool_sizes: List[int] = [1, 2, 4],
        ignore_index: Union[int, None] = None,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_classes (int): Number of possible pixel categories.
            bottom_level (int, optional): Bottom level of inputs this head is attached to. Defaults to 3.
            top_level (int, optional): Top level of inputs this head is attached to. Defaults to 7.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 4.
            ignore_index (Union[int, None], optional): Ignored category index. Defaults to None.
        """
        assert num_classes > 0
        assert len(in_channels) > top_level >= bottom_level > 0
        assert num_channels > 0 and num_layers >= 0
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.pool_sizes = tuple(pool_sizes)
        self.bottom_level = bottom_level
        self.top_level = top_level
        self.ignore_index = ignore_index or -100
        self.levels = list(range(bottom_level, top_level + 1))
        self.rev_levels = list(reversed(range(bottom_level, top_level)))
        self.context_aggregation = SPPM(
            in_channels[top_level], num_channels, self.pool_sizes
        )
        self.lateral_convs = nn.ModuleList(
            [ConvNormAct(in_channels[level], num_channels) for level in self.rev_levels]
        )
        self.upscalers = nn.ModuleList(
            [SimpleUpscaler(num_channels, num_channels) for level in self.rev_levels]
        )
        self.fusions = nn.ModuleList(
            [UAFM(num_channels, num_channels) for level in self.rev_levels]
        )
        self.out_conv = nn.Sequential(
            SequentialConvBlocks(num_channels, num_channels, num_layers),
            nn.Conv2d(num_channels, num_classes, kernel_size=1),
        )
        self.output_shapes = {
            "score_maps": ("batch_size", "height", "width"),
            "class_maps": ("batch_size", "height", "width"),
        }

    def get_logits(self, inputs: List[Tensor]) -> Tensor:
        x = self.context_aggregation(inputs[self.top_level])
        for level, lateral, upscale, fuse in zip(
            self.rev_levels, self.lateral_convs, self.upscalers, self.fusions
        ):
            x = fuse(lateral(inputs[level]), upscale(x))
        return self.out_conv(x)

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        x = interpolate(self.get_logits(inputs), size=inputs[0].shape[2:])
        return x.softmax(dim=1).max(dim=1)

    def training_step(
        self, inputs: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        logits = interpolate(self.get_logits(inputs), size=targets.shape[1:])
        loss = functional.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        return loss, {}

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        metric_kwargs = {
            "task": "multiclass",
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
        }
        self.pixel_accuracy = Accuracy(**metric_kwargs)
        self.mean_iou_computer = JaccardIndex(**metric_kwargs)

    def validation_step(
        self, inputs: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        logits = interpolate(self.get_logits(inputs), size=targets.shape[1:])
        loss = functional.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        scores = logits.softmax(dim=1)
        self.mean_iou_computer.to(loss.device).update(scores, targets)
        self.pixel_accuracy.to(loss.device).update(scores, targets)
        self.loss_computer.to(loss.device).update(loss)
        return loss, {}

    def on_validation_end(self) -> Dict[str, float]:
        return {
            "loss": self.loss_computer.compute().item(),
            "pixel_accuracy": self.pixel_accuracy.compute().item(),
            "mean_iou": self.mean_iou_computer.compute().item(),
        }


class SPPM(nn.Module):
    """https://arxiv.org/abs/2204.02681"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: Tuple[int] = (1, 2, 4),
        with_shortcut: bool = False,
    ) -> None:
        super().__init__()
        self.with_shortcut = with_shortcut
        self.pools = nn.ModuleList(
            [
                nn.Sequential(
                    # HACK: `AdaptiveAvgPool2d` fails ONNX export, so use interpolation
                    BilinearScaler(size=pool_size),
                    ConvNormAct(in_channels, out_channels, 1),
                )
                for pool_size in pool_sizes
            ]
            if len(pool_sizes) > 0
            else [nn.Identity()]
        )
        if with_shortcut:
            self.shortcut = ConvNormAct(in_channels, out_channels, 1)
        self.out_conv = ConvNormAct(out_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        fused = torch.stack(
            [interpolate(pool(x), size=x.shape[2:]) for pool in self.pools]
        ).sum(0)
        if self.with_shortcut:
            fused = fused + self.shortcut(x)
        return self.out_conv(fused)


class UAFM(nn.Module):
    """https://arxiv.org/abs/2204.02681"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvNormAct(4, 1, norm=None, act="sigmoid")

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        alpha = self.conv(
            torch.cat(
                [
                    torch.mean(x1, dim=1, keepdim=True),
                    torch.max(x1, dim=1, keepdim=True).values,
                    torch.mean(x2, dim=1, keepdim=True),
                    torch.max(x2, dim=1, keepdim=True).values,
                ],
                dim=1,
            )
        )
        return x1 * alpha + x2 * (1 - alpha)

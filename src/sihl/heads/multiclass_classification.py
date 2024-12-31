from typing import List, Tuple, Dict

from torch import Tensor, nn
from torch.nn.functional import cross_entropy
from torchmetrics import MeanMetric, Accuracy, Precision, Recall
import torch

from sihl.layers import SequentialConvBlocks


class MulticlassClassification(nn.Module):
    """Multiclass classification is the prediction of the most probable category (or
    "class") for an input image among a finite set of possible categories. One input
    is always matched to exactly one category.
    In the particular case where the set of possible categories is totally ordered, we
    can call this task "ordinal classification"."""

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        num_channels: int = 256,
        num_layers: int = 1,
        level: int = 5,
        label_smoothing: float = 0.0,
        is_ordinal: bool = False,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_classes (int): Number of possible categories.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 1.
            level (int, optional): Level of inputs this head is attached to. Defaults to 5.
            label_smoothing (float, optional): How much to smooth target probability distribution. Defaults to 0.0.
            is_ordinal (bool, optional): Whether the categories are ordered. Defaults to False.
        """
        assert num_classes > 0, num_classes
        assert len(in_channels) > level, (len(in_channels), level)
        assert num_channels > 0 and num_layers > 0, (num_channels, num_layers)
        super().__init__()

        self.num_classes = num_classes
        self.level = level
        self.label_smoothing = label_smoothing
        self.is_ordinal = is_ordinal
        self.convs = nn.Sequential(
            SequentialConvBlocks(in_channels[level], num_channels, num_layers),
            nn.Conv2d(num_channels, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.output_shapes = {
            "scores": ("batch_size", num_classes),
            "classes": ("batch_size",),
        }

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        scores, classes = self.convs(inputs[self.level]).softmax(dim=1).max(dim=1)
        return scores, classes

    def training_step(
        self, inputs: List[Tensor], target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        logits = self.convs(inputs[self.level])
        target = target.to(logits.device)
        if self.is_ordinal:
            target = soft_ordinal_category(target, self.num_classes)
        loss = cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        return loss, {}

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.accuracy_computer = Accuracy("multiclass", num_classes=self.num_classes)
        self.precision_computer = Precision("multiclass", num_classes=self.num_classes)
        self.recall_computer = Recall("multiclass", num_classes=self.num_classes)

    def validation_step(
        self, inputs: List[Tensor], target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        logits = self.convs(inputs[self.level])
        target = target.to(logits.device)
        loss = cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        self.loss_computer.to(loss.device).update(loss)
        self.accuracy_computer.to(logits.device).update(logits, target)
        self.precision_computer.to(logits.device).update(logits, target)
        self.recall_computer.to(logits.device).update(logits, target)
        return loss, {}

    def on_validation_end(self) -> Dict[str, float]:
        return {
            "loss": self.loss_computer.compute().item(),
            "accuracy": self.accuracy_computer.compute().item(),
            "precision": self.precision_computer.compute().item(),
            "recall": self.recall_computer.compute().item(),
        }


def soft_ordinal_category(
    labels: Tensor, num_labels: int, peakiness: float = 1.0
) -> Tensor:
    """https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf"""
    target = torch.arange(num_labels, dtype=torch.float32, device=labels.device)
    return torch.softmax(
        -(target.unsqueeze(0) - labels.unsqueeze(1)).abs() * peakiness, dim=1
    )

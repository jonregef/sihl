from typing import Tuple, List, Dict

from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics import MeanMetric, Accuracy, Precision, Recall
import torch

from sihl.layers import SequentialConvBlocks


class MultilabelClassification(nn.Module):
    """Multilabel classification is the prediction of the subset of all labels relevant
    to an input image. This subset can be empty if no label is relevant. In the
    particular case that the set of all possible labels is a singleton, we can call this
    task "binary classification"."""

    def __init__(
        self,
        in_channels: List[int],
        num_labels: int,
        num_channels: int = 256,
        num_layers: int = 1,
        level: int = 5,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_classes (int): Number of possible labels.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 1.
            level (int, optional): Level of inputs this head is attached to. Defaults to 5.
        """
        assert num_labels > 0, num_labels
        assert len(in_channels) > level, (len(in_channels), level)
        assert num_channels > 0 and num_layers > 0, (num_channels, num_layers)
        super().__init__()

        self.num_labels = num_labels
        self.level = level
        self.convs = nn.Sequential(
            SequentialConvBlocks(in_channels[level], num_channels, num_layers),
            nn.Conv2d(num_channels, num_labels, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.output_shapes = {
            "scores": ("batch_size", num_labels),
            "labels": ("batch_size", num_labels),
        }

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        scores, labels = torch.sort(
            torch.sigmoid(self.convs(inputs[self.level])), descending=True
        )
        return scores, labels

    def training_step(
        self, inputs: List[Tensor], target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        logits = self.convs(inputs[self.level])
        loss = binary_cross_entropy_with_logits(logits, target.to(logits))
        return loss, {}

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.accuracy_computer = Accuracy("multilabel", num_labels=self.num_labels)
        self.precision_computer = Precision("multilabel", num_labels=self.num_labels)
        self.recall_computer = Recall("multilabel", num_labels=self.num_labels)

    def validation_step(
        self, inputs: List[Tensor], target: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        logits = self.convs(inputs[self.level])
        target = target.to(logits)
        loss = binary_cross_entropy_with_logits(logits, target)
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

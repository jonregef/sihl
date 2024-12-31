from typing import Tuple, List, Dict

from torch import nn, Tensor
from torchmetrics import MeanMetric
import torch

from sihl.layers import SequentialConvBlocks


class ViewInvarianceLearning(nn.Module):
    """View invariance learning is a kind of self-supervised learning whereby the model
    is encouraged to produce similar representations for different views (i.e.
    distortions) of the same original image.

    Refs:
        1. [Barlow Twins](https://arxiv.org/abs/2103.03230)
    """

    def __init__(
        self,
        in_channels: List[int],
        embedding_dim: int = 1024,
        level: int = 5,
        num_channels: int = 256,
        num_layers: int = 4,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            embedding_dim (int, optional): Number of embedding channels. Defaults to 1024.
            level (int, optional): Level of inputs this head is attached to. Defaults to 5.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 1.
        """
        assert level < len(in_channels)
        assert num_channels > 0 and num_layers > 0
        super().__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.level = level
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.projector = nn.Sequential(
            SequentialConvBlocks(in_channels[level], num_channels, num_layers),
            nn.Conv2d(num_channels, embedding_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.output_shapes = {"representations": ("batch_size", embedding_dim)}

    def forward(self, inputs: List[Tensor]) -> Tensor:
        return self.projector(inputs[self.level])

    def get_correlation(self, inputs1: List[Tensor], inputs2: List[Tensor]) -> Tensor:
        embedding1 = self.projector(inputs1[self.level])
        if embedding1.shape[0] > 1:
            embedding1 = (embedding1 - embedding1.mean(0)) / embedding1.std(0)
        embedding2 = self.projector(inputs2[self.level])
        if embedding2.shape[0] > 1:
            embedding2 = (embedding2 - embedding2.mean(0)) / embedding2.std(0)
        return torch.mm(embedding1.T, embedding2) / embedding1.shape[0]

    def training_step(
        self, inputs1: List[Tensor], inputs2: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        cross_correlation = self.get_correlation(inputs1, inputs2)
        eye = torch.eye(cross_correlation.shape[0], device=cross_correlation.device)
        invariance = (cross_correlation * eye - eye).pow(2).sum()
        redundancy = (cross_correlation * (1 - eye)).pow(2).sum()
        loss = invariance + redundancy / self.num_channels
        return loss, {}

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.norm_computer = MeanMetric(nan_strategy="ignore")
        self.ondiag_comupter = MeanMetric(nan_strategy="ignore")
        self.offdiag_computer = MeanMetric(nan_strategy="ignore")

    def validation_step(
        self, inputs1: List[Tensor], inputs2: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss = self.training_step(inputs1, inputs2)[0]
        device = loss.device
        self.loss_computer.to(device).update(loss)
        cross_correlation = self.get_correlation(inputs1, inputs2).abs()
        dim = cross_correlation.shape[0]
        identity = torch.eye(dim, device=device)
        norm = torch.linalg.matrix_norm(cross_correlation - identity)
        A_norm = torch.linalg.matrix_norm(cross_correlation)
        I_norm = torch.linalg.matrix_norm(identity)
        max_diff_norm = torch.sqrt(A_norm**2 + I_norm**2)
        self.norm_computer.to(device).update(norm / max_diff_norm)
        mask = identity.to(torch.bool)
        self.ondiag_comupter.to(device).update(cross_correlation[mask].mean())
        self.offdiag_computer.to(device).update(cross_correlation[~mask].mean())
        return loss, {}

    def on_validation_end(self) -> Dict[str, float]:
        return {
            "loss": self.loss_computer.compute().item(),
            "normalized_frobenius_norm": self.norm_computer.compute().item(),
            "on_diagonal_mean": self.ondiag_comupter.compute().item(),
            "off_diagonal_mean": self.offdiag_computer.compute().item(),
        }

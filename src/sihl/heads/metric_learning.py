from typing import List, Tuple, Dict
import math

from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRPrecision
import torch

from sihl.utils import EPS


class MetricLearning(nn.Module):
    """Metric learning is mapping semantically similar images to close points in
    embedding space. It is trained like a multiclass classification head, with pairs of
    sample and corresponding "identities" (equivalent to classes). However, at inference
    time, the metric learning head produces embedding vectors instead of identity
    predictions. These embeddings can then be compared to one another using the eucliean
    distance, enabling clustering, k-NN classification, or other downstream tasks.

    Refs:
        1. [Arcface](https://arxiv.org/abs/1801.07698)

    Todo:
        - [Center Contrastive Loss](https://arxiv.org/abs/2308.00458)
    """

    def __init__(
        self,
        in_channels: List[int],
        num_identities: int,
        embedding_dim: int = 256,
        level: int = 5,
        margin: float = 0.5,
        num_subcenters: int = 1,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_identities (int): Number of possible semantic identities.
            embedding_dim (int, optional): Number of embedding channels. Defaults to 256.
            level (int, optional): Level of inputs this head is attached to. Defaults to 5.
            margin (float, optional): Margin hyperparameter (c.f. [1]). Defaults to 0.5.
            num_subcenters (int, optional): Number of clusters per identity (c.f. [1]). Defaults to 1.
        """
        assert num_identities > 0, num_identities
        assert len(in_channels) > level, (len(in_channels), level)
        assert embedding_dim > 0, embedding_dim
        super().__init__()

        self.num_identities = num_identities
        self.level = level
        self.num_subcenters = num_subcenters
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels[level], embedding_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.scale = math.sqrt(2) * math.log(num_identities - 1)
        self.margin = margin
        self.weight = nn.Parameter(
            torch.FloatTensor(num_subcenters, embedding_dim, num_identities)
        )
        nn.init.xavier_uniform_(self.weight)
        self.index_embeddings, self.index_ids = None, None

        self.output_shapes = {"embeddings": ("batch_size", embedding_dim)}

    def forward(self, inputs: List[Tensor]) -> Tensor:
        return functional.normalize(self.conv(inputs[self.level]))

    def training_step(self, inputs: List[Tensor], targets: Tensor) -> Tensor:
        feats = self.forward(inputs)
        feats = feats.unsqueeze(0).expand(self.num_subcenters, *feats.shape)
        cos_theta = torch.max(
            torch.bmm(feats, functional.normalize(self.weight)), dim=0
        ).values
        theta = torch.acos(torch.clamp(cos_theta, -1 + EPS, 1 - EPS))
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        selected = torch.where(
            theta > math.pi - self.margin, torch.zeros_like(one_hot), one_hot
        ).bool()
        logits = torch.where(selected, theta + self.margin, theta).cos() * self.scale
        return functional.cross_entropy(logits, targets), {}

    def reset_validation_index_set(self) -> None:
        self.index_embeddings = None
        self.index_ids = None

    def extend_validation_index_set(self, inputs: List[Tensor], ids: Tensor) -> None:
        if self.index_embeddings is None:
            self.index_embeddings = self.forward(inputs)
            self.index_ids = ids
        else:
            self.index_embeddings = torch.cat(
                [self.index_embeddings, self.forward(inputs)]
            )
            self.index_ids = torch.cat([self.index_ids, ids])

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        # https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision
        self.precision_computers = {}
        for k in [1, 3, 5]:
            self.precision_computers[k] = RetrievalPrecision(top_k=k)
        self.r_precision = RetrievalRPrecision()
        self.sample_counter = 0
        self.knn_matches = {}
        for k in [1, 3, 5]:
            self.knn_matches[k] = 0

    def validation_step(
        self, inputs: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        embeddings = self.forward(inputs)
        batch_size, device = embeddings.shape[0], embeddings.device
        similarities = functional.linear(embeddings, weight=self.index_embeddings)
        top5_values, top5_targets = torch.topk(
            similarities, k=min(similarities.shape[1], 6)
        )
        top5_targets = self.index_ids[top5_targets]
        indexes = torch.arange(batch_size, device=device).unsqueeze(1)
        preds = top5_values[:, 1:]  # 1:6 instead of 0:5 to remove self similarity
        target = top5_targets[:, 1:] == targets.unsqueeze(1)
        indexes = self.sample_counter + indexes.repeat(1, preds.shape[1])
        for k in [1, 3, 5]:
            self.precision_computers[k].to(device).update(preds, target, indexes)
        self.r_precision.to(device).update(preds, target, indexes)
        self.sample_counter += batch_size
        for k in [1, 3, 5]:
            self.knn_matches[k] += target[:, :k].sum()
        # loss, _ = self.training_step(inputs, targets)  # loss increases ?!
        self.loss_computer.to(device).update(0)
        return torch.zeros((1,)), {}

    def on_validation_end(self) -> Dict[str, float]:
        metrics = {"loss": self.loss_computer.compute().item()}
        for k in [1, 3, 5]:
            metrics[f"precision_at_{k}"] = self.precision_computers[k].compute().item()
        metrics["r_precision"] = self.r_precision.compute().item()
        for k in [1, 3, 5]:
            metrics[f"{k}nn_accuracy"] = self.knn_matches[k] / self.sample_counter / k
        return metrics

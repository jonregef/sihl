from functools import partial
from typing import List, Dict, Tuple, Optional

from einops import rearrange, reduce
from torch import nn, Tensor
from torchmetrics import MeanMetric, JaccardIndex, Accuracy
import torch

from sihl.layers import (
    SimpleDownscaler,
    SimpleUpscaler,
    ConvNormAct,
    BilinearScaler,
    SequentialConvBlocks,
)
from sihl.utils import interpolate, BatchedMeanVarianceAccumulator

sequential_downscalers = partial(SequentialConvBlocks, ConvBlock=SimpleDownscaler)
sequential_upscalers = partial(SequentialConvBlocks, ConvBlock=SimpleUpscaler)


class AnomalyDetection(nn.Module):
    """Anomaly detection is telling whether an image contains an "anomaly" or not. The
    difference with binary classification is that this head only needs "positive samples"
    (i.e. "normal" images) to train on, whereas the binary classification head would
    need those and a similar amount of "negative samples" (i.e. "anomalous" images).
    This task is self-supervised, but it needs labeled samples (positive and negative)
    for validation.

    As described in [1], this head has 3 "submodels":
    1. a pre-trained frozen teacher, which extracts generic image features
    2. an autoencoder, which extracts compact features by attempting to recreate the
       input through a bottleneck
    3. a student model, which tries to match the outputs of the previous 2 submodels

    At prediction time, the difference between the student and the teacher highlights
    structural (local) anomalies, while the difference between the student and the
    autoencoder shows logical (global) anomalies.

    Refs:
        1. [EfficientAD](https://arxiv.org/abs/2303.14535)
    """

    def __init__(
        self,
        in_channels: List[int],
        level: int = 2,
        num_channels: int = 256,
        num_layers: int = 1,
        autoencoder_channels: int = 64,
        autoencoder_top_level: int = 5,
    ):
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            level (int, optional): Top level of inputs this head is attached to. Defaults to 3.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 3.
            autoencoder_channels (int, optional): Number of channels in the compact representation. Defaults to 64.
            autoencoder_top_level (int, optional): Top level of inputs the autoencoder is attached to. Defaults to 5.
        """
        assert num_channels > 0 and num_layers > 0
        assert len(in_channels) > level > 0
        super().__init__()

        self.level = level
        self.num_channels = num_channels
        self.ae_channels = autoencoder_channels
        self.num_layers = num_layers
        self.p_hard = 0.999
        self.autoencoder_top_level = autoencoder_top_level
        self.out_channels = in_channels[level]

        self.student = nn.Sequential(
            ConvNormAct(in_channels[0], num_channels),
            sequential_downscalers(num_channels, num_channels, num_layers=level),
            SequentialConvBlocks(num_channels, num_channels, num_layers=num_layers),
            nn.Conv2d(num_channels, self.out_channels * 2, kernel_size=3, padding=1),
        )

        self.autoencoder_encoder = nn.Sequential(
            ConvNormAct(in_channels[0], self.ae_channels),
            sequential_downscalers(
                self.ae_channels, self.ae_channels, num_layers=autoencoder_top_level
            ),
        )
        size = 8
        self.autoencoder_resize = BilinearScaler(size=(size, size))
        self.autoencoder_bottleneck = nn.Sequential(
            nn.Linear(size * size * self.ae_channels, self.ae_channels),
            nn.Linear(self.ae_channels, size * size * self.ae_channels),
        )
        upscale_levels = autoencoder_top_level - level
        self.autoencoder_decoder = nn.Sequential(
            sequential_upscalers(
                self.ae_channels, self.ae_channels, num_layers=upscale_levels
            ),
            SequentialConvBlocks(
                self.ae_channels, self.ae_channels, num_layers=num_layers
            ),
            nn.Conv2d(self.ae_channels, self.out_channels, kernel_size=3, padding=1),
        )

        self.inputs0, self.inputs_level = [], []
        self.register_buffer("local_thresh", torch.tensor([0.05]))
        self.register_buffer("global_thresh", torch.tensor([0.05]))
        self.register_buffer("features_mean", torch.tensor([0]))
        self.register_buffer("feature_std", torch.tensor([1]))
        self.register_buffer("q_st_start", torch.tensor(0))
        self.register_buffer("q_st_end", torch.tensor(0.1))
        self.register_buffer("q_ae_start", torch.tensor(0))
        self.register_buffer("q_ae_end", torch.tensor(0.1))
        self.output_shapes = {
            "anomaly_maps": ("batch_size", f"height/{2**level}", f"width/{2**level}")
        }

    def compute_distances(self, inputs: List[Tensor]) -> Tensor:
        teacher_out = (inputs[self.level] - self.features_mean) / self.feature_std
        student_out = self.student(inputs[0])

        encoded = self.autoencoder_encoder(inputs[0])
        old_size = encoded.shape[2:]
        encoded = self.autoencoder_resize(encoded)
        h, w = encoded.shape[2:]
        encoded = rearrange(encoded, "b c h w -> b (c h w)")
        encoded = rearrange(
            self.autoencoder_bottleneck(encoded), "b (c h w) -> b c h w", h=h, w=w
        )
        autoencoder_out = self.autoencoder_decoder(interpolate(encoded, size=old_size))

        distance_ae = (autoencoder_out - teacher_out).pow(2)
        distance_st = (teacher_out - student_out[:, : self.out_channels]).pow(2)
        distance_stae = (autoencoder_out - student_out[:, self.out_channels :]).pow(2)
        return distance_st, distance_ae, distance_stae

    def forward(self, inputs: List[Tensor]) -> Tensor:
        distance_st, distance_ae, distance_stae = self.compute_distances(inputs)
        local_anomaly = reduce(distance_st, "b c h w -> b 1 h w", "mean")
        local_anomaly = self.local_thresh * (
            (local_anomaly - self.q_st_start) / (self.q_st_end - self.q_st_start)
        )
        global_anomaly = reduce(distance_stae, "b c h w -> b 1 h w", "mean")
        global_anomaly = self.global_thresh * (
            (global_anomaly - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)
        )
        anomaly = (local_anomaly.relu() + global_anomaly.relu()).clamp(0, 1)
        return interpolate(anomaly, size=inputs[0].shape[2:]).squeeze(1)

    def training_step(
        self,
        inputs: List[Tensor],
        targets: Optional[Tensor] = None,
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        if not is_validating:
            self.inputs0.append(inputs[0])
            self.inputs_level.append(inputs[self.level])
        distance_st, distance_ae, distance_stae = self.compute_distances(inputs)
        loss_st = torch.cat(
            [x[x >= torch.quantile(x, q=self.p_hard)] for x in distance_st]
        ).mean()
        loss_ae = distance_ae.mean()
        loss_stae = distance_stae.mean()
        return loss_st + loss_ae + loss_stae, {
            "loss_student_teacher": loss_st,
            "loss_autoencoder_teacher": loss_ae,
            "loss_student_autoencoder": loss_stae,
        }

    def on_validation_start(self) -> None:
        if len(self.inputs0):
            st_distances, stae_distances = [], []
            for inputs0, inputs_level in zip(self.inputs0, self.inputs_level):
                distance_st, distance_ae, distance_stae = self.compute_distances(
                    [inputs0] + [None for _ in range(self.level - 1)] + [inputs_level]
                )
                st_distances.append(reduce(distance_st, "b c h w -> b h w", "mean"))
                stae_distances.append(reduce(distance_stae, "b c h w -> b h w", "mean"))
            self.inputs0.clear()
            self.inputs_level.clear()
            # https://github.com/pytorch/pytorch/issues/64947
            distance_st = torch.cat(st_distances).flatten()[-(2**24 - 1) :]
            distance_stae = torch.cat(stae_distances).flatten()[-(2**24 - 1) :]
            self.q_st_start = torch.quantile(distance_st, q=0.9)
            self.q_st_end = torch.quantile(distance_st, q=0.995)
            self.q_ae_start = torch.quantile(distance_stae, q=0.9)
            self.q_ae_end = torch.quantile(distance_stae, q=0.995)

        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.mean_iou_computer = JaccardIndex(task="binary")
        self.accuracy = Accuracy(task="binary")

    def validation_step(
        self, inputs: List[Tensor], targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        distance_st, distance_ae, distance_stae = self.compute_distances(inputs)
        loss, metrics = self.training_step(inputs, is_validating=True)
        if targets is not None:
            pred = self.forward(inputs)
            self.mean_iou_computer.to(loss.device).update(pred, targets)
            self.accuracy.to(loss.device).update(
                (pred > 0.5).any(dim=(1, 2)), targets.any(dim=(1, 2))
            )
        self.loss_computer.to(loss.device).update(loss)
        return loss, metrics

    def on_validation_end(self) -> Dict[str, float]:
        return {
            "loss": self.loss_computer.compute().item(),
            "mean_iou": self.mean_iou_computer.compute().item(),
            "accuracy": self.accuracy.compute().item(),
        }

    def on_pretraining_start(self) -> None:
        self.feature_accumulator = BatchedMeanVarianceAccumulator()

    def pretraining_step(
        self, inputs: List[Tensor], targets: Optional[Tensor] = None
    ) -> None:
        features = rearrange(inputs[self.level], "n c h w -> (n h w) c")
        self.feature_accumulator.update(features)

    def on_pretraining_end(self):
        mean, variance = self.feature_accumulator.compute()
        self.features_mean = rearrange(mean, "c -> 1 c 1 1")
        self.feature_std = rearrange(variance.sqrt(), "c -> 1 c 1 1")

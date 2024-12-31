from typing import List, Tuple, Dict, Optional
from functools import partial

from einops.layers.torch import Rearrange
from torch import nn, Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanMetric

from sihl.layers import (
    ConvNormAct,
    SimpleUpscaler,
    SequentialConvBlocks,
    BilinearScaler,
)
from sihl.utils import interpolate

sequential_upscalers = partial(SequentialConvBlocks, ConvBlock=SimpleUpscaler)


class Autoencoding(nn.Module):
    """Autoencoding is reconstructing an image by encoding it into a compact
    representation and then decoding that into the initial input. The point of learning
    this is that the representation can be used for downstream tasks like clustering.
    This task is self-supervised, so it can be a good choice for pre-training backbones.
    """

    def __init__(
        self,
        in_channels: List[int],
        level: int = 5,
        num_channels: int = 256,
        num_layers: int = 3,
        representation_channels: int = 1024,
        prebottleneck_size: Tuple[int, int] = (4, 4),
        activation: Optional[str] = "sigmoid",
    ):
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            level (int, optional): Level of inputs this head is attached to. Defaults to 5.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 1.
            representation_channels (int, optional): Number of channels in the compact representation. Defaults to 128.
            activation (Optional[str], optional): Activation function of the last layer. Defaults to "sigmoid".
        """
        assert num_channels > 0 and num_layers > 0
        assert len(in_channels) > level > 0
        super().__init__()

        self.level = level
        self.size = prebottleneck_size
        self.encoder = nn.Sequential(
            ConvNormAct(in_channels[level], num_channels, 1),
            BilinearScaler(size=self.size),
            Rearrange("b c h w -> b (c h w)", h=self.size[0], w=self.size[1]),
            nn.Linear(
                num_channels * self.size[0] * self.size[1], representation_channels
            ),
            nn.ReLU(),
        )
        self.predecoder = nn.Sequential(
            nn.Linear(
                representation_channels, num_channels * self.size[0] * self.size[1]
            ),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c h w", h=self.size[0], w=self.size[1]),
        )
        self.decoder = nn.Sequential(
            sequential_upscalers(num_channels, num_channels, num_layers=level),
            SequentialConvBlocks(num_channels, num_channels, num_layers=num_layers),
            ConvNormAct(num_channels, in_channels[0], 1, norm=None, act=activation),
        )

        self.output_shapes = {
            "reconstructions": ("batch_size", in_channels[0], "height", "width"),
            "representations": ("batch_size", representation_channels),
        }

    def forward(self, inputs: List[Tensor]) -> Tensor:
        size = inputs[self.level].shape[2:]
        representations = self.encoder(inputs[self.level])
        reconstructions = interpolate(self.predecoder(representations), size=size)
        reconstructions = self.decoder(reconstructions).contiguous()
        return reconstructions, representations

    def training_step(
        self, inputs: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        reconstructions, representations = self.forward(inputs)
        loss = (reconstructions - targets).pow(2).mean()
        return loss, {}

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.mae_computer = MeanAbsoluteError()
        self.mse_computer = MeanSquaredError()

    def validation_step(
        self, inputs: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        reconstructions, representations = self.forward(inputs)
        loss = (reconstructions - targets).pow(2).mean()
        self.loss_computer.to(loss.device).update(loss)
        self.mae_computer.to(loss.device).update(reconstructions, targets)
        self.mse_computer.to(loss.device).update(reconstructions, targets)
        return loss, {}

    def on_validation_end(self) -> Dict[str, float]:
        return {
            "loss": self.loss_computer.compute().item(),
            "mean_absolute_error": self.mae_computer.compute().item(),
            "mean_squared_error": self.mse_computer.compute().item(),
        }

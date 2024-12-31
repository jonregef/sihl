from typing import List, Tuple, Dict

from torch import Tensor, nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanMetric, R2Score

from sihl.layers import SequentialConvBlocks


class Regression(nn.Module):
    """Regression is the prediction of a scalar within a given finite interval."""

    def __init__(
        self,
        in_channels: List[int],
        lower_bound: float,
        upper_bound: float,
        level: int = 5,
        num_channels: int = 256,
        num_layers: int = 1,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            lower_bound (float): Lower bound of the interval of possible values.
            upper_bound (float): Upper bound of the interval of possible values.
            level (int, optional): Level of inputs this head is attached to. Defaults to 5.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 4.
        """
        assert lower_bound < upper_bound
        assert num_channels > 0 and num_layers > 0
        assert level < len(in_channels)
        super().__init__()

        self.convs = nn.Sequential(
            SequentialConvBlocks(in_channels[level], num_channels, num_layers),
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(0),
        )
        self.level = level
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.output_shapes = {"values": ("batch_size",)}

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def denormalize(self, x: Tensor) -> Tensor:
        x = x * (self.upper_bound - self.lower_bound) + self.lower_bound
        return x.clamp(self.lower_bound, self.upper_bound)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        return self.denormalize(self.convs(inputs[self.level]))

    def training_step(
        self, inputs: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        pred_values = self.convs(inputs[self.level])
        return (self.normalize(targets) - pred_values).cosh().log().mean(), {}

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.r2 = R2Score()
        self.mae_computer = MeanAbsoluteError()
        self.mse_computer = MeanSquaredError()

    def validation_step(
        self, inputs: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        pred_values = self.convs(inputs[self.level])
        loss = (self.normalize(targets) - pred_values).cosh().log().mean()
        device = loss.device
        pred_values = self.denormalize(pred_values)
        self.loss_computer.to(device).update(loss)
        self.r2.to(device).update(pred_values, targets)
        self.mae_computer.to(device).update(pred_values, targets)
        self.mse_computer.to(device).update(pred_values, targets)
        return loss, {}

    def on_validation_end(self) -> Dict[str, float]:
        return {
            "loss": self.loss_computer.compute().item(),
            "r_squared": self.r2.compute().item(),
            "mean_absolute_error": self.mae_computer.compute().item(),
            "mean_squared_error": self.mse_computer.compute().item(),
        }

from functools import partial
from typing import List, Tuple, Dict
import math

from einops import rearrange, repeat
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchmetrics.text import EditDistance, WordErrorRate
from torchvision import ops
import torch


class TextRecognition(nn.Module):
    """
    Refs:
        1. (Holistic Representation Guided Attention Network)[https://arxiv.org/abs/1904.01375]
    """

    def __init__(
        self,
        in_channels: List[int],
        num_tokens: int,
        max_sequence_length: int,
        level: int = 3,
        num_channels: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        embedding_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_tokens (int): Number of possible text tokens.
            max_sequence_length (int): Maximum number of tokens to predict in a single sample.
            level (int, optional): Level of inputs this head is attached to. Defaults to 3.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 1.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            embedding_dim (int, optional): Embedding dimensionality for tokens. Defaults to 1024.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        assert num_tokens > 0
        assert max_sequence_length > 0
        assert level < len(in_channels)
        super().__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_tokens = num_tokens
        self.max_sequence_length = max_sequence_length
        self.level = level
        self.pad = num_tokens

        Conv = partial(ops.Conv2dNormActivation, activation_layer=nn.SiLU)
        self.visual_encoding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(in_channels[level], num_channels, kernel_size=1),
        )
        self.lateral_conv = Conv(in_channels[level], num_channels, kernel_size=1)
        self.positional_encoding = PositionalEncoding(
            num_channels, max_len=self.max_sequence_length, dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=num_channels,
                nhead=num_heads,
                dim_feedforward=embedding_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.token_projection = nn.Linear(num_channels, self.num_tokens + 1)
        self.output_shapes = {
            "scores": ("batch_size", max_sequence_length),
            "tokens": ("batch_size", max_sequence_length),
        }

    def forward(self, inputs: List[Tensor]) -> Tensor:
        x = inputs[self.level]
        visual_encoding = repeat(
            self.visual_encoding(x), "b c 1 1 -> b l c", l=self.max_sequence_length
        )
        memory = rearrange(self.lateral_conv(x), "b c h w -> b (h w) c")
        y = self.positional_encoding(visual_encoding)
        logits = self.token_projection(self.decoder(y, memory))
        scores, sequence = torch.max(logits, dim=2)
        return scores, sequence

    def training_step(
        self, inputs: List[Tensor], texts: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        x = inputs[self.level]
        batch_size, device = x.shape[0], x.device
        visual_encoding = repeat(
            self.visual_encoding(x), "b c 1 1 -> b l c", l=self.max_sequence_length
        )
        memory = rearrange(self.lateral_conv(x), "b c h w -> b (h w) c")
        target = torch.full(
            (batch_size, self.max_sequence_length), self.pad, device=device
        )
        for batch_idx, text in enumerate(texts):
            target[batch_idx, : text.shape[0]] = text

        y = self.positional_encoding(visual_encoding)
        logits = rearrange(
            self.token_projection(self.decoder(y, memory)), "b l c -> b c l"
        )
        loss = functional.cross_entropy(logits, target, reduction="none")
        loss = loss.nan_to_num(nan=0).mean()
        return loss, {}

    def on_validation_start(self) -> None:
        self.token_error_rate = WordErrorRate()
        self.edit_distance = EditDistance()
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.matches: List[bool] = []

    def validation_step(
        self, inputs: List[Tensor], tokens: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        scores, pred_texts = self.forward(inputs)
        predictions = [
            " ".join(str(token.item()) for token in text if token != self.pad)
            for text in pred_texts
        ]
        ground_truths = [
            " ".join(str(token.item()) for token in tokens[b])
            for b in range(len(tokens))
        ]
        self.token_error_rate.update(predictions, ground_truths)
        self.matches.extend(
            [pred == gt for pred, gt in zip(predictions, ground_truths)]
        )
        self.edit_distance.update(predictions, ground_truths)
        self.matches.extend(
            [pred == gt for pred, gt in zip(predictions, ground_truths)]
        )
        loss, metrics = self.training_step(inputs, tokens)
        self.loss_computer.to(loss.device).update(loss)
        return loss, metrics

    def on_validation_end(self) -> Dict[str, float]:
        return {
            "loss": self.loss_computer.compute().item(),
            "token_error_rate": self.token_error_rate.compute().item(),
            "edit_distance": self.edit_distance.compute().item(),
            "accuracy": sum(self.matches) / max(len(self.matches), 1),
        }


class PositionalEncoding(nn.Module):
    """https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x shape: (batch_size, seq_len, embedding_dim)"""
        return self.dropout(x + self.pe[:, : x.shape[1], :])

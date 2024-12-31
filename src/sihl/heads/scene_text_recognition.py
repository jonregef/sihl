from typing import List, Tuple, Dict

from einops import rearrange, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchmetrics.text import CharErrorRate, EditDistance
import torch

from sihl.layers import ConvNormAct, SequentialConvBlocks, SimpleUpscaler
from sihl.heads.semantic_segmentation import SPPM, UAFM


class SceneTextRecognition(nn.Module):
    """Scene text recognition is predicting a sequence of tokens. Tokens can represent
    textual characters, but not necessarily. The prediction is performed in parallel,
    which is fast, but can struggle to predict very long sequences.
    """

    def __init__(
        self,
        in_channels: List[int],
        num_tokens: int,
        max_sequence_length: int,
        bottom_level: int = 3,
        top_level: int = 5,
        num_channels: int = 256,
        num_layers: int = 3,
        pool_sizes: List[int] = [1, 2],
        logitnorm_scale: float = 20,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_tokens (int): Number of possible tokens.
            max_sequence_length (int): Maximum length of predicted sequences.
            level (int, optional): Level of inputs this head is attached to. Defaults to 3.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 4.
        """
        assert 0 < num_tokens < 256
        assert max_sequence_length > 0
        assert len(in_channels) > top_level >= bottom_level > 0
        assert num_channels > 0 and num_layers >= 0
        super().__init__()

        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.max_sequence_length = max_sequence_length
        self.pool_sizes = pool_sizes
        self.bottom_level = bottom_level
        self.top_level = top_level
        self.levels = range(bottom_level, top_level + 1)
        self.rev_levels = list(reversed(range(bottom_level, top_level)))
        self.logitnorm_scale = logitnorm_scale
        self.threshold = 0.5

        self.context_aggregation = SPPM(
            in_channels[top_level], num_channels, pool_sizes, with_shortcut=True
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

        self.stem = SequentialConvBlocks(num_channels, num_channels, num_layers)
        self.class_head = nn.Conv2d(num_channels, num_tokens + 1, 1)
        self.index_head = ConvNormAct(
            num_channels, max_sequence_length, 1, norm=None, act="sigmoid"
        )
        self.output_shapes = {
            "scores": ("batch_size", max_sequence_length),
            "tokens": ("batch_size", max_sequence_length),
        }

    def get_features(self, inputs: List[Tensor]) -> Tensor:
        x = self.context_aggregation(inputs[self.top_level])
        for level, lateral, upscale, fuse in zip(
            self.rev_levels, self.lateral_convs, self.upscalers, self.fusions
        ):
            x = fuse(lateral(inputs[level]), upscale(x))
        return self.stem(x)

    def get_maps(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        x = self.get_features(inputs)
        index_map = self.index_head(x)
        class_map = self.class_head(x).softmax(dim=1)
        return index_map, class_map

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        x = self.get_features(inputs)
        index_map = rearrange(self.index_head(x), "n l h w -> n 1 l h w")
        class_map = rearrange(self.class_head(x), "n k h w -> n k 1 h w")
        logits = reduce(class_map * index_map, " n k l h w -> n k l", "mean")
        logits = functional.normalize(logits, dim=1) * self.logitnorm_scale
        scores = logits.softmax(dim=1)
        scores, pred_tokens = scores.max(dim=1)  # (N, L), (N, L)
        paddings = pred_tokens == self.num_tokens
        scores[paddings], pred_tokens[paddings] = 0, 0
        return scores, pred_tokens

    def training_step(
        self, inputs: List[Tensor], tokens: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        x = self.get_features(inputs)
        index_map = rearrange(self.index_head(x), "n l h w -> n 1 l h w")
        class_map = rearrange(self.class_head(x), "n k h w -> n k 1 h w")
        logits = reduce(class_map * index_map, " n k l h w -> n k l", "mean")
        logits = functional.normalize(logits, dim=1) * self.logitnorm_scale
        target_shape = (logits.shape[0], logits.shape[2])
        target = torch.full(target_shape, self.num_tokens, device=logits.device)
        for batch_idx, sample_tokens in enumerate(tokens):
            for char_pos, token_idx in enumerate(sample_tokens):
                target[batch_idx, char_pos] = token_idx
        loss = functional.cross_entropy(logits, target)
        return loss, {}

    def on_validation_start(self) -> None:
        self.token_error_rate = CharErrorRate()
        self.edit_distance = EditDistance()
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.matches: List[bool] = []

    def validation_step(
        self, inputs: List[Tensor], tokens: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        scores, pred_tokens = self.forward(inputs)
        # HACK: using miscellaneous symbols unicode block as 256 unique tokens
        predictions = [
            "".join(
                chr(0x2600 + token.item())
                for score, token in zip(sample_scores, sample_tokens)
                if score > self.threshold and token != self.num_tokens
            ).strip()
            for sample_scores, sample_tokens in zip(scores, pred_tokens)
        ]
        ground_truths = [
            "".join(chr(0x2600 + token.item()) for token in label).strip()
            for label in tokens
        ]
        self.token_error_rate.update(predictions, ground_truths)
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
            "accuracy": sum(self.matches) / len(self.matches),
        }

import math
from typing import Optional, List, Tuple, Dict

from einops import rearrange
from torch import nn, Tensor
from torch.nn import functional
from torch.nn.modules.transformer import _generate_square_subsequent_mask
from torchmetrics.text import WordErrorRate
import torch


class AutoregressiveTextRecognition(nn.Module):
    """
    Autoregressive text recognition is like scene text recognition, but with a
    sequential, autoregressive approach. This means that the latency is higher, but the
    accuracy tends to be better, especially for longer sequences.

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
        dropout: float = 0.0,
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
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        assert num_tokens > 0
        assert max_sequence_length > 0
        assert level < len(in_channels)
        super().__init__()

        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.max_sequence_length = max_sequence_length
        self.level = level
        self.pad, self.begin, self.end = num_tokens, num_tokens + 1, num_tokens + 2

        self.visual_encoding = nn.Sequential(
            nn.Conv2d(in_channels[level], num_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.lateral_conv = nn.Conv2d(in_channels[level], num_channels, kernel_size=1)
        self.embedding = nn.Embedding(self.num_tokens + 3, num_channels, self.pad)
        self.positional_encoding = PositionalEncoding(num_channels, dropout=dropout)
        self.decoder = CausalTransformerDecoder(
            CausalTransformerDecoderLayer(
                d_model=num_channels,
                nhead=num_heads,
                dim_feedforward=embedding_dim,
                dropout=dropout,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(num_channels, self.num_tokens + 3, bias=False)
        # TODO: cloze refinement and/or bidirectional decoding
        self.output_shapes = {"tokens": ("batch_size", max_sequence_length)}

    def get_next_token(
        self,
        sequence: Tensor,
        visual_encoding: Tensor,
        memory: Tensor,
        cache: Optional[Tensor],
    ) -> Tensor:
        sequence = sequence.to(torch.int64)
        y = self.positional_encoding(self.embedding(sequence) + visual_encoding)
        output, cache = self.decoder(y, memory, cache)
        output = self.fc(output)
        pred_tokens = torch.argmax(output, dim=2)[-1, :]
        pred_tokens[sequence[-1, :] == self.end] = self.end
        sequence = torch.cat([sequence, pred_tokens.unsqueeze(0)])
        return sequence, visual_encoding, memory, cache

    def forward(self, inputs: List[Tensor]) -> Tensor:
        x = inputs[self.level]
        visual_encoding = self.visual_encoding(x).unsqueeze(0)
        batch_size, device = x.shape[0], visual_encoding.device
        memory = rearrange(self.lateral_conv(x), "b c h w -> (h w) b c")
        sequence = torch.full((1, batch_size), self.begin, device=device)
        cache = None
        for idx in range(1, self.max_sequence_length + 1):
            sequence, visual_encoding, memory, cache = self.get_next_token(
                sequence, visual_encoding, memory, cache
            )
            # if torch.all(sequence[-1, :] == self.end):  # FIXME: ONNX export
            #     break
        sequence[sequence == self.begin] = self.pad
        sequence[sequence == self.end] = self.pad
        sequence = rearrange(sequence[1:], "l b -> b l")  # [1:] removes [BEGIN]
        return sequence

    def training_step(
        self, inputs: List[Tensor], texts: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        x = inputs[self.level]
        visual_encoding = self.visual_encoding(x).unsqueeze(0)
        memory = rearrange(self.lateral_conv(x), "b c h w -> (h w) b c")

        target = pad_tensors(texts, pad=self.end).to(torch.int64).to(memory.device)
        target = functional.pad(target, (0, 1), value=self.end)  # (N, S)
        sequence = functional.pad(target[:, :-1], (1, 0), value=self.begin)  # (N, S)

        y = self.positional_encoding(self.embedding(sequence.t()) + visual_encoding)
        logits = self.fc(self.decoder.training_step(y, memory)).permute(1, 2, 0)
        loss = functional.cross_entropy(logits, target, reduction="mean")
        return loss, {}

    @torch.no_grad()
    def on_validation_start(self) -> None:
        self.token_error_rate = WordErrorRate()
        self.matches: List[bool] = []

    @torch.no_grad()
    def validation_step(
        self, inputs: List[Tensor], texts: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        pred_texts = self.forward(inputs)
        predictions = [
            " ".join(str(token.item()) for token in text if token != self.pad)
            for text in pred_texts
        ]
        ground_truths = [
            " ".join(str(token.item()) for token in text) for text in texts
        ]
        self.token_error_rate.update(predictions, ground_truths)
        self.matches.extend(
            [pred == gt for pred, gt in zip(predictions, ground_truths)]
        )
        return self.training_step(inputs, texts)

    @torch.no_grad()
    def on_validation_end(self) -> Dict[str, float]:
        return {
            "token_error_rate": self.token_error_rate.compute().item(),
            "accuracy": sum(self.matches) / max(len(self.matches), 1),
        }


class PositionalEncoding(nn.Module):
    """https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x shape: (seq_len, batch_size, embedding_dim)"""
        return self.dropout(x + self.pe[: x.shape[0]])


class CausalTransformerDecoder(nn.TransformerDecoder):
    """https://github.com/alex-matton/causal-transformer-decoder"""

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """tgt shape: (seq_len x batch_size, embedding_dim)"""
        output = tgt

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output])

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache)

        return output, new_cache

    def training_step(self, tgt: Tensor, memory: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod.training_step(output, memory)
        return output


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt: Tensor, memory: Optional[Tensor] = None) -> Tensor:
        tgt_last_tok = tgt[-1:, :, :]
        tgt_last_tok = self.norm1(tgt_last_tok)  # NOTE: pre-LN
        tmp_tgt = self.self_attn(tgt_last_tok, tgt, tgt, attn_mask=None)[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)

        if memory is not None:
            tgt_last_tok = self.norm2(tgt_last_tok)
            # TODO: get attention maps for visualization
            tmp_tgt = self.multihead_attn(tgt_last_tok, memory, memory)[0]
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)

        tgt_last_tok = self.norm3(tgt_last_tok)
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        return tgt_last_tok

    def training_step(self, tgt: Tensor, memory: Optional[Tensor] = None) -> Tensor:
        return super().forward(
            tgt,
            memory,
            tgt_mask=_generate_square_subsequent_mask(tgt.size(0), tgt.device),
        )


def pad_tensors(tensors: List[Tensor], pad: int) -> Tensor:
    assert len(tensors)
    max_length = max(len(t) for t in tensors)
    padded = [
        functional.pad(t.unsqueeze(0), (0, max_length - len(t)), value=pad).squeeze(0)
        for t in tensors
    ]
    return torch.stack(padded)

from typing import List
from pathlib import Path

from torch import Tensor
import onnx
import onnxruntime
import pytest
import torch

from sihl.layers import BiFPN


BATCH_SIZE, TOP_CHANNELS, TOP_HEIGHT, TOP_WIDTH = 4, 1024, 16, 16
OUT_CHANNELS = 64
NUM_CLASSES, NUM_OBJECTS = 10, 5
BOTTOM_LEVEL, TOP_LEVEL = 3, 7
ONNX_FILE_NAME, ONNX_VERSION = "bifpn.onnx", 18


@pytest.fixture()
def model() -> BiFPN:
    return BiFPN(
        in_channels=[3] + [TOP_CHANNELS // 2**_ for _ in range(5)][::-1],
        out_channels=OUT_CHANNELS,
        bottom_level=BOTTOM_LEVEL,
        top_level=TOP_LEVEL,
    )


@pytest.fixture()
def backbone_output() -> List[Tensor]:
    return [
        torch.randn(BATCH_SIZE, 3, TOP_HEIGHT * 32, TOP_WIDTH * 32),
        torch.randn(BATCH_SIZE, TOP_CHANNELS // 16, TOP_HEIGHT * 16, TOP_WIDTH * 16),
        torch.randn(BATCH_SIZE, TOP_CHANNELS // 8, TOP_HEIGHT * 8, TOP_WIDTH * 8),
        torch.randn(BATCH_SIZE, TOP_CHANNELS // 4, TOP_HEIGHT * 4, TOP_WIDTH * 4),
        torch.randn(BATCH_SIZE, TOP_CHANNELS // 2, TOP_HEIGHT * 2, TOP_WIDTH * 2),
        torch.randn(BATCH_SIZE, TOP_CHANNELS, TOP_HEIGHT, TOP_WIDTH),
    ]


def test_forward(model: BiFPN, backbone_output: List[Tensor]) -> None:
    outputs = model(backbone_output)
    assert len(outputs) == TOP_LEVEL + 1
    for output in outputs[BOTTOM_LEVEL : TOP_LEVEL + 1]:
        assert output.shape[1] == OUT_CHANNELS


def test_levels(backbone_output: List[Tensor]) -> None:
    model = BiFPN(
        in_channels=[3] + [TOP_CHANNELS // 2**_ for _ in range(5)][::-1],
        out_channels=OUT_CHANNELS,
        bottom_level=len(backbone_output) - 3,
        top_level=len(backbone_output) - 2,
    )
    outputs = model(backbone_output)
    assert len(outputs) == len(backbone_output)
    assert outputs[len(backbone_output) - 1].shape[1] == backbone_output[-1].shape[1]
    assert outputs[len(backbone_output) - 2].shape[1] == OUT_CHANNELS
    assert outputs[len(backbone_output) - 3].shape[1] == OUT_CHANNELS


@pytest.fixture()
def onnx_model(model: BiFPN, backbone_output: List[Tensor]) -> None:
    torch.onnx.export(
        model,
        backbone_output,
        ONNX_FILE_NAME,
        opset_version=ONNX_VERSION,
        input_names=[f"input_level_{idx}" for idx in range(len(backbone_output))],
    )

    onnx_model = onnx.load(ONNX_FILE_NAME)
    Path(ONNX_FILE_NAME).unlink()
    return onnx_model


def test_onnx_inference(
    onnx_model, model: BiFPN, backbone_output: List[Tensor]
) -> None:
    model.eval()
    with torch.no_grad():
        pytorch_output = model(backbone_output)
    onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    onnx_input = {
        f"input_level_{idx}": _.numpy()
        for idx, _ in enumerate(backbone_output)
        if f"input_level_{idx}" in [node.name for node in onnx_model.graph.input]
    }
    onnx_output = onnx_session.run(None, onnx_input)
    for idx in range(len(onnx_output)):
        torch.testing.assert_close(pytorch_output[idx].numpy(), onnx_output[idx])

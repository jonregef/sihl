from pathlib import Path
from typing import List

from torch import Tensor
import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from sihl.heads import DepthEstimation

BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH = 5, 256, 128, 128
BOTTOM_LEVEL, TOP_LEVEL = 3, 5
LOWER_BOUND, UPPER_BOUND = 0, 10
ONNX_VERSION, ONNX_FILE_NAME = 18, "depth_estimation.onnx"


@pytest.fixture()
def model() -> DepthEstimation:
    return DepthEstimation(
        in_channels=[3] + [NUM_CHANNELS for _ in range(TOP_LEVEL)],
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        bottom_level=BOTTOM_LEVEL,
        top_level=TOP_LEVEL,
    )


@pytest.fixture()
def backbone_output() -> List[Tensor]:
    return [torch.randn((BATCH_SIZE, 3, HEIGHT, WIDTH))] + [
        torch.randn((BATCH_SIZE, NUM_CHANNELS, HEIGHT // 2**_, WIDTH // 2**_))
        for _ in range(1, TOP_LEVEL + 1)
    ]


@pytest.fixture()
def targets() -> Tensor:
    return (
        torch.rand((BATCH_SIZE, HEIGHT, WIDTH)) * (UPPER_BOUND - LOWER_BOUND)
        + LOWER_BOUND
    )


def test_forward(model: DepthEstimation, backbone_output: List[Tensor]) -> None:
    anomaly_map = model(backbone_output)
    assert tuple(anomaly_map.shape) == (BATCH_SIZE, HEIGHT, WIDTH)


def test_training_step(
    model: DepthEstimation, backbone_output: List[Tensor], targets: Tensor
) -> None:
    loss, _ = model.training_step(backbone_output, targets, masks=targets > 0)
    assert loss.item()


def test_validation_step(
    model: DepthEstimation, backbone_output: List[Tensor], targets: Tensor
) -> None:
    model.on_validation_start()
    loss, _ = model.validation_step(backbone_output, targets, masks=targets > 0)
    assert loss.item()
    metrics = model.on_validation_end()
    assert metrics


@pytest.fixture()
def onnx_model(model: DepthEstimation, backbone_output: List[Tensor]) -> None:
    torch.onnx.export(
        model.to(torch.float32),
        args=backbone_output,
        f=ONNX_FILE_NAME,
        opset_version=ONNX_VERSION,
        input_names=[f"input_level_{idx}" for idx in range(len(backbone_output))],
        output_names=[f"head0/{name}" for name, shape in model.output_shapes.items()],
        dynamic_axes=dict(
            {
                f"input_level_{lvl}": {
                    0: "batch_size",
                    2: f"height/{2**lvl}",
                    3: f"width/{2**lvl}",
                }
                for lvl in range(len(backbone_output))
            },
            **{
                f"head0/{name}": {
                    shape_idx: str(shape_value)
                    for shape_idx, shape_value in enumerate(shape)
                }
                for name, shape in model.output_shapes.items()
            },
        ),
        external_data=False,
        verify=True,
        # dynamo=True,
        # report=True,
    )
    onnx_model = onnx.load(ONNX_FILE_NAME)
    Path(ONNX_FILE_NAME).unlink()
    return onnx_model


def test_onnx_inference(
    onnx_model, model: DepthEstimation, backbone_output: List[Tensor]
) -> None:
    model.eval()
    onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    onnx_input = {
        f"input_level_{idx}": _.numpy()
        for idx, _ in enumerate(backbone_output)
        if f"input_level_{idx}" in [node.name for node in onnx_model.graph.input]
    }
    # just check that 99% of values are equal.
    pytorch_output = model(backbone_output).detach().numpy()
    onnx_output = onnx_session.run(None, onnx_input)[0]
    assert np.sum(np.abs(pytorch_output - onnx_output) > 1) / onnx_output.size < 0.01

from typing import List, Dict
from pathlib import Path

from torch import Tensor
import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from sihl.heads import PanopticSegmentation

BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH = 5, 256, 128, 128
BOTTOM_LEVEL, TOP_LEVEL = 3, 7
ONNX_VERSION, ONNX_FILE_NAME = 17, "panoptic_segmentation.onnx"
MAX_INSTANCES = 100
NUM_THING_CLASSES, NUM_STUFF_CLASSES = 16, 8


@pytest.fixture()
def model() -> PanopticSegmentation:
    return PanopticSegmentation(
        in_channels=[3] + [NUM_CHANNELS for _ in range(TOP_LEVEL)],
        num_stuff_classes=NUM_STUFF_CLASSES,
        num_thing_classes=NUM_THING_CLASSES,
        bottom_level=BOTTOM_LEVEL,
        top_level=TOP_LEVEL,
        max_instances=MAX_INSTANCES,
    )


@pytest.fixture()
def backbone_output() -> List[Tensor]:
    return [torch.rand((BATCH_SIZE, 3, HEIGHT, WIDTH))] + [
        torch.rand((BATCH_SIZE, NUM_CHANNELS, HEIGHT // 2**_, WIDTH // 2**_))
        for _ in range(1, TOP_LEVEL + 1)
    ]


@pytest.fixture()
def targets() -> Dict[str, List[Tensor]]:
    return torch.stack(
        [
            torch.cat(
                [
                    torch.randint(
                        0, NUM_THING_CLASSES + NUM_STUFF_CLASSES, (1, HEIGHT, WIDTH)
                    ),
                    torch.randint(0, MAX_INSTANCES, (1, HEIGHT, WIDTH)),
                ],
            )
            for _ in range(BATCH_SIZE)
        ]
    )


def test_forward(model: PanopticSegmentation, backbone_output: List[Tensor]) -> None:
    panoptic_map = model(backbone_output)
    assert tuple(panoptic_map.shape) == (BATCH_SIZE, 2, HEIGHT, WIDTH)


def test_training_step(
    model: PanopticSegmentation, backbone_output: List[Tensor], targets: List[List[str]]
) -> None:
    loss, _ = model.training_step(backbone_output, targets)
    assert loss.item()


def test_validation_step(
    model: PanopticSegmentation, backbone_output: List[Tensor], targets: List[List[str]]
) -> None:
    model.on_validation_start()
    loss, _ = model.validation_step(backbone_output, targets)
    assert loss.item()
    metrics = model.on_validation_end()
    assert metrics


@pytest.fixture()
def onnx_model(model: PanopticSegmentation, backbone_output: List[Tensor]) -> None:
    torch.onnx.export(
        model,
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
    onnx_model, model: PanopticSegmentation, backbone_output: List[Tensor]
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

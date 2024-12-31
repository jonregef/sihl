from typing import List, Dict
from pathlib import Path

from torch import Tensor
import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from sihl.heads import InstanceSegmentation

BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH = 5, 256, 128, 128
BOTTOM_LEVEL, TOP_LEVEL = 3, 7
NUM_CLASSES = 16
ONNX_VERSION, ONNX_FILE_NAME = 18, "instance_segmentation.onnx"


@pytest.fixture()
def model() -> InstanceSegmentation:
    return InstanceSegmentation(
        in_channels=[3] + [NUM_CHANNELS for _ in range(TOP_LEVEL)],
        num_classes=NUM_CLASSES,
        bottom_level=BOTTOM_LEVEL,
        top_level=TOP_LEVEL,
    )


@pytest.fixture()
def backbone_output() -> List[Tensor]:
    return [torch.rand((BATCH_SIZE, 3, HEIGHT, WIDTH))] + [
        torch.rand((BATCH_SIZE, NUM_CHANNELS, HEIGHT // 2**_, WIDTH // 2**_))
        for _ in range(1, TOP_LEVEL + 1)
    ]


@pytest.fixture()
def targets() -> Dict[str, List[Tensor]]:
    return {
        "classes": [torch.randint(0, NUM_CLASSES, (_,)) for _ in range(BATCH_SIZE)],
        "masks": [
            torch.randint(0, 2, (_, HEIGHT, WIDTH), dtype=torch.float)
            for _ in range(BATCH_SIZE)
        ],
    }


def test_forward(model: InstanceSegmentation, backbone_output: List[Tensor]) -> None:
    num_instances, scores, pred_classes, pred_masks = model(backbone_output)
    assert tuple(num_instances.shape) == (BATCH_SIZE,)
    assert tuple(scores.shape) == (BATCH_SIZE, model.max_instances)
    assert tuple(pred_classes.shape) == (BATCH_SIZE, model.max_instances)
    assert tuple(pred_masks.shape) == (BATCH_SIZE, model.max_instances, HEIGHT, WIDTH)


def test_training_step(
    model: InstanceSegmentation, backbone_output: List[Tensor], targets: List[List[str]]
) -> None:
    loss, _ = model.training_step(backbone_output, **targets)
    assert loss.item()


def test_validation_step(
    model: InstanceSegmentation, backbone_output: List[Tensor], targets: List[List[str]]
) -> None:
    model.on_validation_start()
    loss, _ = model.validation_step(backbone_output, **targets)
    assert loss.item()
    metrics = model.on_validation_end()
    assert metrics


@pytest.fixture()
def onnx_model(model: InstanceSegmentation, backbone_output: List[Tensor]) -> None:
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
    onnx_model, model: InstanceSegmentation, backbone_output: List[Tensor]
) -> None:
    model.eval()
    onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    onnx_input = {
        f"input_level_{idx}": _.numpy()
        for idx, _ in enumerate(backbone_output)
        if f"input_level_{idx}" in [node.name for node in onnx_model.graph.input]
    }
    # just check that 99% of values are equal.
    pytorch_output = [_.detach().numpy() for _ in model(backbone_output)]
    onnx_output = onnx_session.run(None, onnx_input)
    for i in range(len(pytorch_output)):
        assert (
            np.sum(np.abs(pytorch_output[i] - onnx_output[i]) > 1) / onnx_output[i].size
            < 0.01
        )

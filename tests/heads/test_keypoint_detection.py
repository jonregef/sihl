from typing import List, Dict
from pathlib import Path

from torch import Tensor
import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from sihl.heads import KeypointDetection

BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH = 5, 256, 128, 128
MAX_INSTANCES, BOTTOM_LEVEL, TOP_LEVEL = 100, 3, 7
NUM_KEYPOINTS = 8
ONNX_VERSION, ONNX_FILE_NAME = 18, "keypoint_detection.onnx"


@pytest.fixture()
def model() -> KeypointDetection:
    return KeypointDetection(
        in_channels=[3] + [NUM_CHANNELS for _ in range(TOP_LEVEL)],
        num_keypoints=NUM_KEYPOINTS,
        bottom_level=BOTTOM_LEVEL,
        top_level=TOP_LEVEL,
        max_instances=MAX_INSTANCES,
    )


@pytest.fixture()
def backbone_output() -> List[Tensor]:
    return [torch.randn((BATCH_SIZE, 3, HEIGHT, WIDTH))] + [
        torch.randn((BATCH_SIZE, NUM_CHANNELS, HEIGHT // 2**_, WIDTH // 2**_))
        for _ in range(1, TOP_LEVEL + 1)
    ]


@pytest.fixture()
def targets() -> Dict[str, List[Tensor]]:
    num_objects = range(BATCH_SIZE)  # tests target with 0 objects too
    return {
        "presence": [
            torch.randint(0, 2, (num_objects[_], NUM_KEYPOINTS), dtype=bool)
            for _ in range(BATCH_SIZE)
        ],
        "keypoints": [
            torch.rand((num_objects[_], NUM_KEYPOINTS, 2)) for _ in range(BATCH_SIZE)
        ],
    }


def test_forward(model: KeypointDetection, backbone_output: List[Tensor]) -> None:
    num_instances, scores, keypoint_scores, pred_keypoints = model(backbone_output)
    assert tuple(num_instances.shape) == (BATCH_SIZE,)
    assert tuple(scores.shape) == (BATCH_SIZE, model.max_instances)
    assert tuple(keypoint_scores.shape) == (
        BATCH_SIZE,
        model.max_instances,
        NUM_KEYPOINTS,
    )
    assert tuple(pred_keypoints.shape) == (
        BATCH_SIZE,
        model.max_instances,
        NUM_KEYPOINTS,
        2,
    )


def test_training_step(
    model: KeypointDetection, backbone_output: List[Tensor], targets: List[List[str]]
) -> None:
    loss, _ = model.training_step(backbone_output, **targets)
    assert loss.item() >= 0


def test_validation_step(
    model: KeypointDetection, backbone_output: List[Tensor], targets: List[List[str]]
) -> None:
    model.on_validation_start()
    loss, _ = model.validation_step(backbone_output, **targets)
    assert loss.item() >= 0
    metrics = model.on_validation_end()
    assert metrics


@pytest.fixture()
def onnx_model(model: KeypointDetection, backbone_output: List[Tensor]) -> None:
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
    onnx_model, model: KeypointDetection, backbone_output: List[Tensor]
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

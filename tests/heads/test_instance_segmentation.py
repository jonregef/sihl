from pathlib import Path
from typing import List, Dict

from onnxslim import slim
from torch import Tensor
from torch.export import Dim
import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from sihl.heads import InstanceSegmentation

BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH = 8, 256, 128, 128
BOTTOM_LEVEL, TOP_LEVEL = 3, 7
NUM_CLASSES = 16
ONNX_VERSION, ONNX_FILE_NAME = 20, "instance_segmentation.onnx"


@pytest.fixture()
def model() -> InstanceSegmentation:
    return InstanceSegmentation(
        in_channels=[3] + [NUM_CHANNELS for _ in range(TOP_LEVEL)],
        num_classes=NUM_CLASSES,
        bottom_level=BOTTOM_LEVEL,
        top_level=TOP_LEVEL,
    ).eval()


@pytest.fixture()
def backbone_output() -> List[Tensor]:
    return [torch.rand((BATCH_SIZE, 3, HEIGHT, WIDTH))] + [
        torch.rand((BATCH_SIZE, NUM_CHANNELS, HEIGHT // 2**_, WIDTH // 2**_))
        for _ in range(1, TOP_LEVEL + 1)
    ]


@pytest.fixture()
def targets() -> Dict[str, List[Tensor]]:
    num_objects = range(BATCH_SIZE)  # tests target with 0 objects too
    return {
        "classes": [
            torch.randint(0, NUM_CLASSES, (num_objects[_],)) for _ in range(BATCH_SIZE)
        ],
        "masks": [
            torch.randint(0, 2, (num_objects[_], HEIGHT, WIDTH), dtype=torch.float32)
            for _ in range(BATCH_SIZE)
        ],
    }


def test_forward(model: InstanceSegmentation, backbone_output: List[Tensor]) -> None:
    num_instances, scores, pred_classes, pred_masks = model.forward(backbone_output)
    assert tuple(num_instances.shape) == (BATCH_SIZE,)
    assert tuple(scores.shape) == (BATCH_SIZE, model.max_instances)
    assert tuple(pred_classes.shape) == (BATCH_SIZE, model.max_instances)
    assert tuple(pred_masks.shape) == (BATCH_SIZE, model.max_instances, HEIGHT, WIDTH)


def test_training_step(
    model: InstanceSegmentation,
    backbone_output: List[Tensor],
    targets: Dict[str, List[Tensor]],
) -> None:
    loss, _ = model.training_step(backbone_output, **targets)
    assert loss.item() >= 0


def test_validation_step(
    model: InstanceSegmentation,
    backbone_output: List[Tensor],
    targets: Dict[str, List[Tensor]],
) -> None:
    model.on_validation_start()
    loss, _ = model.validation_step(backbone_output, **targets)
    assert loss.item() >= 0
    metrics = model.on_validation_end()
    assert metrics


@pytest.fixture()
def onnx_model(model: InstanceSegmentation, backbone_output: List[Tensor]) -> None:
    batch_size, height, width = Dim("batch_size"), Dim("height"), Dim("width")
    torch.onnx.export(
        model.eval().to(torch.float32),
        args=(backbone_output,),
        f=ONNX_FILE_NAME,
        opset_version=ONNX_VERSION,
        output_names=model.output_shapes.keys(),
        dynamic_shapes=(
            [  # FIXME: dynamic shape doesn't work
                (batch_size, Dim.STATIC, Dim.STATIC, Dim.STATIC)
                for level in range(len(backbone_output))
            ],
        ),
        dynamo=True,
        external_data=False,
        verify=True,
        # report=True,
    )
    onnx_model = onnx.load(ONNX_FILE_NAME)
    onnx_model = slim(onnx_model)
    onnx.save(onnx_model, ONNX_FILE_NAME)
    Path(ONNX_FILE_NAME).unlink()
    return onnx_model


def test_onnx_inference(
    onnx_model, model: InstanceSegmentation, backbone_output: List[Tensor]
) -> None:
    model.eval()
    onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    in_names = [_.name for _ in onnx_model.graph.input]
    onnx_input = {
        f"inputs_{idx}": x.numpy()
        for idx, x in enumerate(backbone_output)
        if f"inputs_{idx}" in in_names
    }
    # just check that 99% of values are equal.
    pytorch_output = [_.detach().numpy() for _ in model.forward(backbone_output)]
    onnx_output = onnx_session.run(None, onnx_input)
    for i in range(len(pytorch_output)):
        assert (
            np.sum(np.abs(pytorch_output[i] - onnx_output[i]) > 1) / onnx_output[i].size
            < 0.01
        )

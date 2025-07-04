from pathlib import Path
from typing import List, Tuple

from torch import Tensor
import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from sihl.heads import QuadrilateralDetection


BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH = 4, 256, 128, 128
NUM_CLASSES, MIN_LEVEL, MAX_LEVEL = 100, 3, 7
ONNX_VERSION, ONNX_FILE_NAME = 18, "quadrilateral_detection.onnx"


@pytest.fixture()
def model() -> QuadrilateralDetection:
    return QuadrilateralDetection(
        in_channels=(3,) + (NUM_CHANNELS,) * MAX_LEVEL, num_classes=NUM_CLASSES
    )


@pytest.fixture()
def backbone_output() -> List[Tensor]:
    return [torch.randn((BATCH_SIZE, 3, HEIGHT, WIDTH))] + [
        torch.randn((BATCH_SIZE, NUM_CHANNELS, HEIGHT // 2**_, WIDTH // 2**_))
        for _ in range(MAX_LEVEL)
    ]


@pytest.fixture()
def targets() -> Tuple[List[Tensor], List[Tensor]]:
    NUM_OBJECTS = 10
    return (
        [torch.randint(0, NUM_CLASSES, (NUM_OBJECTS,)) for _ in range(BATCH_SIZE)],
        [
            torch.randint(0, HEIGHT, (NUM_OBJECTS, 4, 2), dtype=torch.float)
            for _ in range(BATCH_SIZE)
        ],
    )


def test_forward(model: QuadrilateralDetection, backbone_output: List[Tensor]) -> None:
    num_instances, scores, classes, quads = model(backbone_output)
    assert tuple(num_instances.shape) == (BATCH_SIZE,)
    assert tuple(scores.shape) == (BATCH_SIZE, model.max_instances)
    assert tuple(classes.shape) == (BATCH_SIZE, model.max_instances)
    assert tuple(quads.shape) == (BATCH_SIZE, model.max_instances, 4, 2)


def test_training_step(
    model: QuadrilateralDetection,
    backbone_output: List[Tensor],
    targets: Tuple[List[Tensor], List[Tensor]],
) -> None:
    loss, _ = model.training_step(backbone_output, *targets)
    assert loss.item()


# def test_validation_step(
#     model: QuadrilateralDetection,
#     backbone_output: List[Tensor],
#     targets: Tuple[List[Tensor], List[Tensor]],
# ) -> None:
#     model.on_validation_start()
#     loss, _ = model.validation_step(backbone_output, *targets)
#     assert loss.item()
#     metrics = model.on_validation_end()
#     assert metrics


# @pytest.fixture()
# def onnx_model(model: QuadrilateralDetection, backbone_output: List[Tensor]) -> None:
#     torch.onnx.export(
#         model.to(torch.float32),
#         args=backbone_output,
#         f=ONNX_FILE_NAME,
#         opset_version=ONNX_VERSION,
#         input_names=[f"input_level_{idx}" for idx in range(len(backbone_output))],
#         output_names=[f"head0/{name}" for name, shape in model.output_shapes.items()],
#         dynamic_axes=dict(
#             {
#                 f"input_level_{lvl}": {
#                     0: "batch_size",
#                     2: f"height/{2**lvl}",
#                     3: f"width/{2**lvl}",
#                 }
#                 for lvl in range(len(backbone_output))
#             },
#             **{
#                 f"head0/{name}": {
#                     shape_idx: str(shape_value)
#                     for shape_idx, shape_value in enumerate(shape)
#                 }
#                 for name, shape in model.output_shapes.items()
#             },
#         ),
#         external_data=False,
#         verify=True,
#         # dynamo=True,
#         # report=True,
#     )
#     onnx_model = onnx.load(ONNX_FILE_NAME)
#     Path(ONNX_FILE_NAME).unlink()
#     return onnx_model


# def test_onnx_inference(
#     onnx_model, model: QuadrilateralDetection, backbone_output: List[Tensor]
# ) -> None:
#     model.eval()
#     onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
#     onnx_input = {
#         f"input_level_{idx}": _.numpy()
#         for idx, _ in enumerate(backbone_output)
#         if f"input_level_{idx}" in [node.name for node in onnx_model.graph.input]
#     }
#     # just check that 99% of values are equal.
#     pytorch_output = [_.detach().numpy() for _ in model(backbone_output)]
#     onnx_output = onnx_session.run(None, onnx_input)
#     for i in range(len(pytorch_output)):
#         assert (
#             np.sum(np.abs(pytorch_output[i] - onnx_output[i]) > 1) / onnx_output[i].size
#             < 0.01
#         )

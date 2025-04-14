# Simple Image Heads and Layers

[![PyPI](https://img.shields.io/pypi/v/sihl.svg)][pypi_]
[![python versions](https://img.shields.io/pypi/pyversions/sihl)][python version]
[![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jonregef/c203d6bce2a485ab49d1814ff3218a06/raw/covbadge.json)][coverage]

[pypi_]: https://pypi.org/project/sihl/
[python version]: https://pypi.org/project/sihl
[coverage]: https://coverage.readthedocs.io/en/7.2.5/

Pytorch implementations of computer vision tasks that aim to be readable, efficient, and effective.

Most of the code is based on published research, adapted to be easy to understand and use, sometimes at the cost of decreased benchmark performance compared to official figures.

`pip install sihl` to get started. Check out the [examples](./examples/README.md).

## Models

Models have a backbone (from [torchvision](./src/sihl/torchvision_backbone.py) or [timm](./src/sihl/timm_backbone.py)), an optional neck ([FPN](./src/sihl/layers/fpn.py) or [BiFPN](./src/sihl/layers/bifpn.py)), and one or more heads (enabling multitask learning).

Each head corresponds to a task:

- [Anomaly detection](./examples/anomaly_detection.py)
- [Autoencoding](./examples/autoencoding.py)
- [Autoregressive text recognition](./examples/autoregressive_text_recognition.py)
- [Depth estimation](./examples/depth_estimation.py)
- [Instance segmentation](./examples/instance_segmentation.py)
- [Keypoint detection](./examples/keypoint_detection.py)
- [Metric learning](./examples/metric_learning.py)
- [Multiclass classification](./examples/multiclass_classification.py)
- [Multilabel classification](./examples/multilabel_classification.py)
- [Object detection](./examples/object_detection.py)
- [Panoptic segmentation](./examples/panoptic_segmentation.py)
- [Quadrilateral detection](./examples/quadrilateral_detection.py)
- [Regression](./examples/regression.py)
- [Scene text recognition](./examples/scene_text_recognition.py)
- [Semantic segmentation](./examples/semantic_segmentation.py)
- [View invariance learning](./examples/view_invariance_learning.py)

## Development

We recommend using [rye](https://rye.astral.sh/) to manage this project:

- Set your preferred python version with `rye pin 3.X` (3.9 or later).
- If you have a local nvidia GPU, run examples with: `rye run python examples/[...].py`.
- See generated logs with `rye run tensorboard --logdir examples/logs/[...]`.
- Run tests with `rye run pytest tests/`.

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

- [Anomaly detection](./src/sihl/heads/anomaly_detection.py)
- [Autoencoding](./src/sihl/heads/autoencoding.py)
- [Autoregressive text recognition](./src/sihl/heads/autoregressive_text_recognition.py)
- [Depth estimation](./src/sihl/heads/depth_estimation.py)
- [Instance segmentation](./src/sihl/heads/instance_segmentation.py)
- [Keypoint detection](./src/sihl/heads/keypoint_detection.py)
- [Metric learning](./src/sihl/heads/metric_learning.py)
- [Multiclass classification](./src/sihl/heads/multiclass_classification.py)
- [Multilabel classification](./src/sihl/heads/multilabel_classification.py)
- [Object detection](./src/sihl/heads/object_detection.py)
- [Panoptic segmentation](./src/sihl/heads/panoptic_segmentation.py)
- [Quadrilateral detection](./src/sihl/heads/quadrilateral_detection.py)
- [Regression](./src/sihl/heads/regression.py)
- [Scene text recognition](./src/sihl/heads/scene_text_recognition.py)
- [Semantic segmentation](./src/sihl/heads/semantic_segmentation.py)
- [View invariance learning](./src/sihl/heads/view_invariance_learning.py)

## Development

We recommend using [rye](https://rye.astral.sh/) to manage this project:

- Set your preferred python version with `rye pin 3.X` (3.9 or later).
- If you have a local GPU, run examples with: `rye run python examples/[...].py`.
- See generated logs with `rye run tensorboard --logdir examples/logs/[...]`.
- Run tests with `rye run pytest tests/`.

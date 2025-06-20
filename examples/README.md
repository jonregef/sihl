# Examples

`rye run python examples/[...].py` to run an example (from the project's root directory).

Datasets are downloaded into `examples/data/`, and tensorboard logs are saved to `examples/logs/`.

These example training scripts assume availability of a local nvidia GPU, and valid kaggle credentials to download datasets.

All examples use the [imagenet-pretrained Resnet50 backbone from torchvision](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights).

You should get results similar to these:

| task | steps | dataset | metric name | best value |
| :--- | :---: | :---: | :---: | :---: |
| [anomaly detection](./anomaly_detection.py) | 300 | MVTEC cable | accuracy | 0.72 |
| [autoencoding](./autoencoding.py) | 200 | Stanford Cars | mean absolute error | 0.06 |
| [depth estimation](./depth_estimation.py) | 30 | NYU V2 | mean absolute error | 0.35 |
| [instance segmentation](./instance_segmentation.py) | 90k | COCO 2017 | mean average precision (mask) | 27% |
| [keypoint detection](./keypoint_detection.py) | 90k | COCO 2017 | percent of correct keypoint | 68% |
| [metric learning](./metric_learning.py) | 5 | Stanford Cars | nearest neighbor accuracy | 0.79 |
| [multiclass classification](./multiclass_classification.py) | 50 | Stanford Cars | accuracy | 88% |
| [multilabel classification](./multilabel_classification.py) | 12 | COCO 2017 | accuracy | 0.98 |
| [object detection](./object_detection.py) | 90k | COCO 2017 | mean average precision (box) | 35% |
| [panoptic segmentation](./panoptic_segmentation.py) | 12 | COCO 2017 | (box) mean average precision | 0.29 |
| [quadrilateral detection](./quadrilateral_detection.py) | 185 | Military aircrafts | mean average precision | 66% |
| [regression](./regression.py) | 10 | Age prediction | mean absolute error | 6 |
| [semantic segmentation](./semantic_segmentation.py) | 12 | COCO 2017 | mean IOU | 0.28 |
| [text recognition](./text_regression.py) | 100 | Cyrillic text | accuracy | 43% |
| [view invariance learning](./view_invariance_learning.py) | 30 | Stanford Cars | Froebenius norm | 0.58 |

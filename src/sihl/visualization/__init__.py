# ruff: noqa: F401, F811
from typing import Any

from torch import Tensor

from sihl.sihl_model import SihlModel
from sihl.heads import ViewInvarianceLearning

from .common import get_images
from . import anomaly_detection
from . import autoencoding
from . import autoregressive_text_recognition
from . import depth_estimation
from . import instance_segmentation
from . import keypoint_detection
from . import metric_learning
from . import multiclass_classification
from . import multilabel_classification
from . import object_detection
from . import panoptic_segmentation
from . import quadrilateral_detection
from . import regression
from . import scene_text_recognition
from . import semantic_segmentation
from . import view_invariance_learning


def visualize(
    model: SihlModel,
    configs: Any,
    input: Tensor,
    targets: Any,
    logger: Any,
    step: int,
    start_idx: int = 0,
):
    features = model.eval().extract_features(input)
    for idx, (config, head, target) in enumerate(zip(configs, model.heads, targets)):
        if isinstance(head, ViewInvarianceLearning):  # FIXME: this is a hack
            target = model.extract_features(target[: features[0].shape[0]])
        vizs = get_images(head, config, input, target, features)  # singly dispatched
        for viz_idx, viz_img in enumerate(vizs):
            logger.experiment.add_image(
                f"{idx}/visualizations/{start_idx + viz_idx}", viz_img, global_step=step
            )

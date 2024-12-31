# ruff: noqa: F401
from typing import Protocol, Any, List, Dict, Tuple, Union

from torch import Tensor

from sihl.heads.anomaly_detection import AnomalyDetection
from sihl.heads.autoencoding import Autoencoding
from sihl.heads.autoregressive_text_recognition import AutoregressiveTextRecognition
from sihl.heads.depth_estimation import DepthEstimation
from sihl.heads.instance_segmentation import InstanceSegmentation
from sihl.heads.keypoint_detection import KeypointDetection
from sihl.heads.metric_learning import MetricLearning
from sihl.heads.multiclass_classification import MulticlassClassification
from sihl.heads.multilabel_classification import MultilabelClassification
from sihl.heads.object_detection import ObjectDetection
from sihl.heads.panoptic_segmentation import PanopticSegmentation
from sihl.heads.quadrilateral_detection import QuadrilateralDetection
from sihl.heads.regression import Regression
from sihl.heads.scene_text_recognition import SceneTextRecognition
from sihl.heads.semantic_segmentation import SemanticSegmentation
from sihl.heads.view_invariance_learning import ViewInvarianceLearning


TensorShape = Tuple[Union[str, int], ...]
Loss = Tensor
Metrics = Dict[str, float]


class Head(Protocol):
    output_shapes: Dict[str, TensorShape]

    def forward(self, inputs: List[Tensor]) -> Any:
        """Performs an inference pass.
        This function is traced and converted when exporting to ONNX.

        Args:
            inputs (List[Tensor]): Input tensors by feature level

        Returns:
            Any: Depends on the head
        """
        pass

    def training_step(self, inputs: List[Tensor], *args: Any) -> Tuple[Loss, Metrics]:
        pass

    def on_validation_start(self) -> None:
        pass

    def validation_step(self, inputs: List[Tensor], *args: Any) -> Tuple[Loss, Metrics]:
        pass

    def on_validation_end(self) -> Metrics:
        pass

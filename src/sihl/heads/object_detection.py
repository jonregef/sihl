from typing import Any, Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torchmetrics import MeanMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou, complete_box_iou_loss
import torch

from sihl.layers import SequentialConvBlocks, ConvNormAct
from sihl.utils import coordinate_grid, f_score, EPS, focal_loss, interpolate


class ObjectDetection(nn.Module):
    """Object detection is the prediction of the set of "objects" (pairs of axis-aligned
    rectangular bounding boxes and the corresponding category) in the input image.

    Refs:
        1. [One-to-Few Label Assignment for End-to-End Dense Detection](https://arxiv.org/abs/2303.11567)
        2. [TOOD: Task-aligned One-stage Object Detection](https://arxiv.org/abs/2108.07755)
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        bottom_level: int = 3,
        top_level: int = 7,
        num_channels: int = 256,
        num_layers: int = 4,
        max_instances: int = 100,
        t_min: float = 0.2,
        t_max: float = 0.6,
        topk: int = 7,
        soft_label_decay_steps: int = 1,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_classes (int): Number of possible object categories.
            bottom_level (int, optional): Bottom level of inputs this head is attached to. Defaults to 3.
            top_level (int, optional): Top level of inputs this head is attached to. Defaults to 7.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 4.
            max_instances (int, optional): Maximum number of instances to predict in a sample. Defaults to 100.
            t_min (float, optional): Lower bound of O2F parameter t. Defaults to 0.2.
            t_max (float, optional): Upper bound of O2F parameter t. Defaults to 0.6.
            topk (int, optional): How many anchors to match to each ground truth object when copmuting the loss. Defaults to 7.
            soft_label_decay_steps (int, optional): How many training steps to perform before the one-to-few matching becomes one-to-one. Defaults to 1.
        """
        assert num_classes > 0, num_classes
        assert len(in_channels) > top_level, (len(in_channels), top_level)
        assert 0 < bottom_level <= top_level, (bottom_level, top_level)
        assert num_channels > 0 and num_layers > 0, (num_channels, num_layers)
        assert max_instances > 0, max_instances
        assert 0 <= t_min <= t_max <= 1, (t_min, t_max)
        assert topk > 0 and soft_label_decay_steps > 0, (topk, soft_label_decay_steps)
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bottom_level = bottom_level
        self.top_level = top_level
        self.levels = range(bottom_level, top_level + 1)
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.max_instances = max_instances
        self.t_min, self.t_max, self.topk = t_min, t_max, topk
        self.soft_label_decay_steps, self.current_step = soft_label_decay_steps, 0
        self.register_buffer("threshold", torch.tensor(0.5))  # adjusted in validation
        self.box_iou = box_iou  # enables overriding these in children classes
        self.box_loss = complete_box_iou_loss

        # input channels need to be matched if they aren't already
        matched_in_channels = in_channels[bottom_level]
        self.lateral_convs = None
        if not all(in_channels[lvl] == matched_in_channels for lvl in self.levels):
            matched_in_channels = num_channels
            self.lateral_convs = [
                ConvNormAct(in_channels[lvl], num_channels, 1) for lvl in self.levels
            ]

        self.regression_stem = SequentialConvBlocks(
            matched_in_channels, num_channels, num_layers=num_layers, norm="group"
        )
        self.box_head = ConvNormAct(num_channels, 4, 1, norm=None, act="softplus")
        self.class_head = nn.Sequential(
            SequentialConvBlocks(
                matched_in_channels, num_channels, num_layers=num_layers, norm="group"
            ),
            ConvNormAct(num_channels, num_classes, 1, norm=None, act="sigmoid"),
        )
        self.centerness_head = nn.Sequential(
            SequentialConvBlocks(
                num_channels, num_channels // 4, num_layers=num_layers - 1, norm="group"
            ),
            ConvNormAct(num_channels // 4, 1, kernel_size=1, norm=None, act="sigmoid"),
        )
        self.output_shapes = {
            "num_instances": ("batch_size",),
            "scores": ("batch_size", max_instances),
            "classes": ("batch_size", max_instances),
            "boxes": ("batch_size", max_instances, 4),
        }

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scores, reg_features = self.get_features(inputs)
        scores, pred_classes = scores.max(dim=2)
        pred_boxes = self.get_boxes(reg_features)
        batch_size = pred_boxes.shape[0]
        batches = repeat(torch.arange(batch_size), f"b -> b {self.max_instances}")
        instance_idxs = scores.topk(self.max_instances).indices
        scores = scores[batches, instance_idxs]
        pred_classes = pred_classes[batches, instance_idxs]
        pred_boxes = pred_boxes[batches, instance_idxs]
        num_instances = reduce(scores > self.threshold, "b i -> b", "sum")
        return num_instances, scores, pred_classes, pred_boxes

    def get_features(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        torch._assert(len(inputs) > self.top_level, "too few input levels")
        xs = [inputs[level] for level in self.levels]
        if self.lateral_convs is not None:
            xs = [lat(x) for lat, x in zip(self.lateral_convs, xs)]
        reg_features = [self.regression_stem(x) for x in xs]
        centers = [self.centerness_head(reg_feat) for reg_feat in reg_features]
        scores = [self.class_head(x) for x in xs]
        scores = [
            rearrange(s * c, "b c h w -> b (h w) c") for s, c in zip(scores, centers)
        ]
        scores = torch.cat(scores, dim=1)
        return scores, reg_features

    def get_saliency(self, inputs: List[Tensor]) -> Tensor:
        torch._assert(len(inputs) > self.top_level, "too few input levels")
        xs = [inputs[level] for level in self.levels]
        if self.lateral_convs is not None:
            xs = [lat(x) for lat, x in zip(self.lateral_convs, xs)]
        reg_features = [self.regression_stem(x) for x in xs]
        centers = [self.centerness_head(x) for x in reg_features]
        scores = [self.class_head(x) for x in xs]
        size = xs[0].shape[2:]
        scores = [
            interpolate(reduce(s * c, "b c h w -> b 1 h w", "max"), size=size)
            for s, c in zip(scores, centers)
        ]
        return reduce(torch.cat(scores, dim=1), "b c h w -> b h w", "mean")

    def get_boxes(self, features: List[Tensor]) -> Tensor:
        boxes = []
        for level, x in zip(self.levels, features):
            stride, (h, w) = 2**level, x.shape[2:]
            biases = coordinate_grid(h, w, h * stride, w * stride).to(x)
            biases = repeat(biases, "c h w -> (2 c) h w", c=2)
            biases = rearrange(biases, "c h w -> 1 (h w) c", c=4)
            level_boxes = rearrange(self.box_head(x), "b c h w -> b (h w) c", c=4)
            scale = stride * torch.tensor([[[-1, -1, 1, 1]]]).to(x)
            boxes.append(biases + level_boxes * scale)
        return torch.cat(boxes, dim=1)

    def inside_box(self, points: Tensor, box: Tensor) -> Tensor:
        mask = (points[:, 0] > box[0]) & (points[:, 0] < box[2])
        return mask & (points[:, 1] > box[1]) & (points[:, 1] < box[3])

    @torch.no_grad()
    def get_targets(
        self,
        anchors: Tensor,
        scores: Tensor,
        pred_boxes: Tensor,
        gt_classes: List[Tensor],
        gt_boxes: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        progress = min(max(0, self.current_step / self.soft_label_decay_steps), 1)
        ambiguous_samples_factor = (self.t_min - self.t_max) * progress + self.t_max
        if not is_validating:
            self.current_step += 1
        (batch_size, num_instances, _), device = scores.shape, scores.device
        class_target = torch.zeros_like(scores)
        target_shape = (batch_size, num_instances)
        assignment = torch.zeros(target_shape, device=device, dtype=torch.long)
        box_target = torch.zeros(target_shape + pred_boxes.shape[2:], device=device)
        for batch_idx in range(batch_size):
            for gt_idx in range(gt_boxes[batch_idx].shape[0]):
                gt_box = gt_boxes[batch_idx][gt_idx]
                gt_class = gt_classes[batch_idx][gt_idx]
                # only consider predictions whose anchor lie inside the gt box
                mask = self.inside_box(anchors, gt_box)
                mask = mask.nonzero(as_tuple=True)[0].to(device)
                if mask.numel() == 0:  # can happen if the gt box is too small
                    continue
                candidate_ious = self.box_iou(
                    gt_box.unsqueeze(0), pred_boxes[batch_idx, mask]
                )[0]
                candidate_scores = scores[batch_idx, mask, gt_class]
                with torch.autocast(device_type="cuda", enabled=False):
                    candidate_scores = candidate_scores.to(torch.float32)
                    quality = candidate_scores.pow(0.2) * candidate_ious.pow(0.8)
                quality = quality.nan_to_num(0).to(scores)  # where are the nans from?
                topk_values, topk_idxs = quality.topk(k=min(quality.numel(), self.topk))
                if len(topk_values) > 1 and topk_values[1] > EPS:
                    topk_values = topk_values.to(scores) / topk_values[1]
                    topk_values = topk_values * ambiguous_samples_factor
                topk_values[0] = 1.0  # "certain" anchor is at index 0 (it's sorted)
                topk_idxs = mask[topk_idxs]
                for pos_idx, pos_value in zip(topk_idxs, topk_values):
                    if pos_value < EPS:
                        continue
                    if class_target[batch_idx, pos_idx].max() > pos_value:
                        continue
                    class_target[batch_idx, pos_idx, gt_class] = pos_value
                    assignment[batch_idx, pos_idx] = gt_idx
                    box_target[batch_idx, pos_idx] = gt_box
        return class_target, box_target, assignment

    def get_anchors(self, inputs: List[Tensor], normalized: bool = False) -> Tensor:
        anchors_by_level = []
        for level in self.levels:
            height, width = inputs[level].shape[2:]
            y_max = 1.0 if normalized else height * 2**level
            x_max = 1.0 if normalized else width * 2**level
            coords = coordinate_grid(height, width, y_max, x_max).to(inputs[level])
            anchors_by_level.append(rearrange(coords, "c h w -> (h w) c", c=2))
        return torch.cat(anchors_by_level)

    def get_losses(
        self,
        scores: Tensor,
        pred_boxes: Tensor,
        class_target: Tensor,
        box_target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        pos_mask = reduce(class_target, "b i s -> b i", "max") > EPS
        if not pos_mask.any():
            device = scores.device
            return torch.zeros(1, device=device), torch.zeros(1, device=device)
        with torch.autocast(device_type="cuda", enabled=False):
            scores = scores.to(torch.float32).nan_to_num(nan=0.0)
            class_target = class_target.to(torch.float32)
            class_loss = focal_loss(scores, class_target).sum() / class_target.sum()
            box_target = box_target.to(torch.float32)
            box_loss = self.box_loss(
                pred_boxes[pos_mask], box_target[pos_mask], reduction="mean"
            )
        return class_loss, box_loss

    def training_step(
        self,
        inputs: List[Tensor],
        classes: List[Tensor],
        boxes: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        scores, reg_features = self.get_features(inputs)
        pred_boxes = self.get_boxes(reg_features)
        class_target, box_target, assignment = self.get_targets(
            self.get_anchors(inputs), scores, pred_boxes, classes, boxes, is_validating
        )
        class_loss, box_loss = self.get_losses(
            scores, pred_boxes, class_target, box_target
        )
        return class_loss + box_loss, {"class_loss": class_loss, "box_loss": box_loss}

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.map_computer = MeanAveragePrecision(
            extended_summary=True, backend="faster_coco_eval"
        )

    def validation_step(
        self, inputs: List[Tensor], classes: List[Tensor], boxes: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        num_instances, scores, pred_classes, pred_boxes = self.forward(inputs)
        self.map_computer.to(pred_boxes.device).update(
            [
                {"scores": s, "labels": c, "boxes": b}
                for s, c, b in zip(scores, pred_classes, pred_boxes)
            ],
            [{"labels": c, "boxes": b} for c, b in zip(classes, boxes)],
        )
        loss, metrics = self.training_step(inputs, classes, boxes, is_validating=True)
        self.loss_computer.to(loss.device).update(loss)
        return loss, metrics

    def on_validation_end(self) -> Dict[str, Any]:
        metrics = self.map_computer.compute()
        precision = metrics["precision"][0, :, :, 0, 2].mean(dim=1)
        recall = torch.linspace(0.0, 1.0, round(1.0 / 0.01) + 1)
        scores = metrics["scores"][0, :, :, 0, 2].mean(dim=1)
        f1 = f_score(1.0)(precision, recall)
        best_idx = int(f1.argmax().item())
        self.threshold = scores[best_idx]
        metrics["threshold"] = self.threshold
        metrics["precision"] = precision[best_idx]
        metrics["recall"] = recall[best_idx]
        metrics["f1"] = f1[best_idx]
        metrics["f0.5"] = f_score(0.5)(precision[best_idx], recall[best_idx])
        metrics["f2"] = f_score(2.0)(precision[best_idx], recall[best_idx])
        metrics["loss"] = self.loss_computer.compute()
        del metrics["map_per_class"], metrics["mar_100_per_class"]
        del metrics["scores"], metrics["classes"], metrics["ious"]
        return {k: v.to(self.map_computer.device) for k, v in metrics.items()}

from functools import partial
from typing import Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import ops
import torch


class ObjectDetection(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        bottom_level: int = 3,
        top_level: int = 5,
        num_channels: int = 256,
        num_layers: int = 4,
        max_instances: int = 100,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_classes (int): Number of possible object categories.
            bottom_level (int, optional): Bottom level of inputs this head is attached to. Defaults to 3.
            top_level (int, optional): Top level of inputs this head is attached to. Defaults to 5.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 4.
            max_instances (int, optional): Maximum number of instances to predict in a sample. Defaults to 100.
        """
        assert num_classes > 0, num_classes
        assert len(in_channels) > top_level, (len(in_channels), top_level)
        assert 0 < bottom_level <= top_level, (bottom_level, top_level)
        assert num_channels % 4 == 0, num_channels
        assert num_layers >= 0, num_layers
        assert max_instances > 0, max_instances
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bottom_level, self.top_level = bottom_level, top_level
        self.levels = range(bottom_level, top_level + 1)
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.max_instances = max_instances
        self.topk = 9

        MLP = partial(ops.MLP, norm_layer=nn.LayerNorm, activation_layer=nn.SiLU)
        Conv = partial(ops.Conv2dNormActivation, activation_layer=None)
        self.laterals = nn.ModuleList(
            [Conv(in_channels[level], num_channels, 1) for level in self.levels]
        )
        hidden_channels = [num_channels] * num_layers
        self.loc_head = MLP(num_channels, hidden_channels + [1])
        self.loc_head[-2].bias.data.fill_(-5.0)  # bias for low values originally
        self.cls_head = MLP(num_channels, hidden_channels + [num_classes])
        self.box_head = MLP(num_channels, hidden_channels + [4])  # [x1, y1, x2, y2]
        self.iou_head = MLP(num_channels, hidden_channels + [1])  # training only

        self.output_shapes = {
            "num_instances": ("batch_size",),
            "scores": ("batch_size", max_instances),
            "classes": ("batch_size", max_instances),
            "boxes": ("batch_size", max_instances, 4),
        }

    def get_saliency(self, inputs: List[Tensor]) -> Tensor:
        device = inputs[self.bottom_level].device
        batch_size, _, full_height, full_width = inputs[self.bottom_level].shape
        output = torch.zeros((batch_size, full_height, full_width), device=device)
        for lateral, level in zip(self.laterals, self.levels):
            height, width = inputs[level].shape[2:]
            feats = rearrange(lateral(inputs[level]), "b c h w -> b (h w) c")
            scores = self.loc_head(feats).sigmoid()
            scores = rearrange(scores, "b (h w) c -> b c h w", h=height, w=width, c=1)
            scores = functional.interpolate(scores, size=(full_height, full_width))
            output = torch.maximum(output, scores.squeeze(1))
        return output

    def get_offsets_and_scales(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        device = inputs[0].device
        offsets, scales = [], []
        for level in range(self.bottom_level, self.top_level + 1):
            h, w = inputs[level].shape[2:]
            y_min, x_min = 1 / h / 2, 1 / w / 2
            ys = torch.linspace(y_min, 1 - y_min, steps=h, device=device)
            xs = torch.linspace(x_min, 1 - x_min, steps=w, device=device)
            coordinate_grid = torch.stack(
                [repeat(xs, "w -> h w", h=h), repeat(ys, "h -> h w", w=w)], dim=2
            )
            offsets.append(repeat(coordinate_grid, "h w c -> (h w) (2 c)"))
            cell_bbox = torch.tensor([-x_min, -y_min, x_min, y_min], device=device)
            scales.append(repeat(cell_bbox, "c -> i c", i=h * w))
        return torch.cat(offsets), torch.cat(scales)

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (batch_size, _, height, width), device = inputs[0].shape, inputs[0].device
        full_size = torch.tensor([[[width, height, width, height]]], device=device)
        feats = [
            lateral(inputs[level]) for level, lateral in zip(self.levels, self.laterals)
        ]
        flat_feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], 1)
        offsets, scales = self.get_offsets_and_scales(inputs)
        # compute locations
        loc_logits = self.loc_head(flat_feats).squeeze(2)
        loc_logits, loc_idxs = loc_logits.topk(self.max_instances, dim=1)
        batches = repeat(torch.arange(batch_size), f"b -> b {self.max_instances}")

        flat_feats = flat_feats[batches, loc_idxs]
        scores = loc_logits.sigmoid()
        num_instances = reduce(scores > 0.5, "b i -> b", "sum")
        # compute classes
        class_logits = self.cls_head(flat_feats)
        classes = class_logits.max(dim=2).indices
        # compute boxes
        offsets = repeat(offsets, "i c -> b i c", b=batch_size)[batches, loc_idxs]
        scales = repeat(scales, "i c -> b i c", b=batch_size)[batches, loc_idxs]
        box_preds = (offsets + scales * self.box_head(flat_feats).exp()) * full_size
        return num_instances, scores, classes, box_preds

    def training_step(
        self,
        inputs: List[Tensor],
        classes: List[Tensor],
        boxes: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert len(inputs) > self.top_level, "too few input levels"
        device = inputs[0].device
        batch_size, _, full_height, full_width = inputs[0].shape
        full_size = torch.tensor(
            [[full_width, full_height, full_width, full_height]], device=device
        )

        # compute anchors
        offsets, scales = self.get_offsets_and_scales(inputs)
        anchors = (offsets + scales) * full_size

        # assign anchors to gt objects
        matching_results = [
            self.bbox_matching(anchors, boxes[b], self.topk, relative=True)
            for b in range(batch_size)
        ]
        assignment = torch.stack([_[0] for _ in matching_results])
        rel_iou = torch.stack([_[1] for _ in matching_results])

        # compute features
        feats = [
            lateral(inputs[level]) for level, lateral in zip(self.levels, self.laterals)
        ]
        flat_feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], 1)

        # location loss
        loc_logits = self.loc_head(flat_feats).squeeze(2)
        with torch.autocast(device_type="cuda", enabled=False):
            loc_target = (rel_iou == 1.0).to(torch.float32)
            loc_loss = functional.binary_cross_entropy_with_logits(
                loc_logits, loc_target, reduction="none"
            )
            loc_loss = loc_loss.sum() / loc_target.sum()

        if rel_iou.max() == 0:
            metrics = {
                "location_loss": loc_loss,
                "box_loss": torch.zeros_like(loc_loss),
                "class_loss": torch.zeros_like(loc_loss),
                "iou_loss": torch.zeros_like(loc_loss),
            }
            return loc_loss, metrics

        # iou loss
        iou_preds = self.iou_head(flat_feats).squeeze(2)
        with torch.autocast(device_type="cuda", enabled=False):
            iou_loss = functional.mse_loss(
                iou_preds.to(torch.float32), rel_iou, reduction="none"
            )
            iou_loss = iou_loss.sum() / rel_iou.sum()

        o2m_mask = rel_iou > 0
        o2m_weights = rel_iou[o2m_mask]
        o2m_feats = flat_feats[o2m_mask]

        # box loss
        offsets = torch.cat([offsets[mask] for mask in o2m_mask])
        scales = torch.cat([scales[mask] for mask in o2m_mask])
        box_preds = offsets + scales * self.box_head(o2m_feats).exp()
        box_target = torch.cat(
            [boxes[b][assignment[b, m]] for b, m in enumerate(o2m_mask)]
        )
        with torch.autocast(device_type="cuda", enabled=False):
            box_loss = ops.complete_box_iou_loss(
                box_preds, box_target.to(torch.float32) / full_size, reduction="none"
            )
            box_loss = (o2m_weights * box_loss).sum() / o2m_weights.sum()

        # classification loss
        class_logits = self.cls_head(o2m_feats)
        class_target = torch.cat(
            [classes[b][assignment[b, m]] for b, m in enumerate(o2m_mask)]
        )
        with torch.autocast(device_type="cuda", enabled=False):
            class_loss = functional.cross_entropy(
                class_logits.to(torch.float32), class_target, reduction="none"
            )
            class_loss = (o2m_weights * class_loss).sum() / o2m_weights.sum()

        loss = loc_loss + 10 * box_loss + class_loss + iou_loss
        metrics = {
            "location_loss": loc_loss,
            "box_loss": box_loss,
            "class_loss": class_loss,
            "iou_loss": iou_loss,
        }
        return loss, metrics

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        max_detection_thresholds = [1, min(self.max_instances, 10), self.max_instances]
        self.map_computer = MeanAveragePrecision(
            max_detection_thresholds=max_detection_thresholds,
            backend="faster_coco_eval",
        )

    def validation_step(
        self, inputs: List[Tensor], classes: List[Tensor], boxes: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        num_instances, scores, pred_classes, pred_boxes = self.forward(inputs)
        self.map_computer.to(scores.device).update(
            [
                {"scores": s, "labels": c, "boxes": b}
                for s, c, b in zip(scores, pred_classes, pred_boxes)
            ],
            [{"labels": c, "boxes": b} for c, b in zip(classes, boxes)],
        )
        loss, metrics = self.training_step(inputs, classes, boxes, is_validating=True)
        self.loss_computer.to(loss.device).update(loss)
        return loss, metrics

    def on_validation_end(self) -> Dict[str, float]:
        metrics = {}
        if hasattr(self, "map_computer"):
            metrics = self.map_computer.compute()
            for key in list(metrics.keys()):
                if "per_class" in key or key in {"classes", "ious"}:
                    del metrics[key]
        metrics["loss"] = self.loss_computer.compute()
        return metrics

    @staticmethod
    def bbox_matching(
        anchors: Tensor, gt_boxes: Tensor, topk: int, relative: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        device = anchors.device
        num_anchors, num_gt = anchors.shape[0], gt_boxes.shape[0]
        o2m_assignments = torch.full((num_anchors,), -1, device=device)  # gt idx or -1
        o2m_iou = torch.zeros((num_anchors,), device=device)
        if num_gt == 0:
            return o2m_assignments, o2m_iou

        ious = ops.complete_box_iou(anchors, gt_boxes).clamp(0)
        topk_ious, topk_idxs = torch.topk(ious, k=topk, dim=0)
        is_best_match = torch.zeros((num_anchors, num_gt), dtype=bool, device=device)
        is_best_match.scatter_(0, topk_idxs[0:1], True)
        is_topk_match = torch.zeros((num_anchors, num_gt), dtype=bool, device=device)
        is_topk_match.scatter_(0, topk_idxs, True)

        max_ious, max_gt_idxs = torch.max(ious * is_topk_match.float(), dim=1)
        valid_mask = is_topk_match.any(dim=1)
        o2m_assignments[valid_mask] = max_gt_idxs[valid_mask]
        o2m_iou[valid_mask] = max_ious[valid_mask]

        if relative:
            o2m_rel_iou = torch.zeros((num_anchors,), device=device)
            best_ious_per_gt = topk_ious[0]
            best_iou_for_assignment = best_ious_per_gt[max_gt_idxs]
            o2m_rel_iou[valid_mask] = (
                max_ious[valid_mask] / best_iou_for_assignment[valid_mask]
            ).nan_to_num(0)
            return o2m_assignments, o2m_rel_iou
        else:
            return o2m_assignments, o2m_iou

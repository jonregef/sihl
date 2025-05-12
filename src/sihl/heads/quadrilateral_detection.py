from functools import partial
from typing import Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import ops
import torch


class QuadrilateralDetection(nn.Module):
    """Quadrilateral detection is the prediction of the set of "objects" (pairs of
    convex quadrilaterals and the corresponding category) in the input image.
    """

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
            top_level (int, optional): Top level of inputs this head is attached to. Defaults to 7.
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
        Conv = partial(ops.Conv2dNormActivation, activation_layer=nn.SiLU)
        self.laterals = nn.ModuleList(
            [Conv(in_channels[level], num_channels, 1) for level in self.levels]
        )
        self.global_context = nn.Sequential(
            Conv(in_channels[self.top_level], num_channels, 1), nn.AdaptiveAvgPool2d(1)
        )
        hidden_channels = [num_channels] * num_layers
        self.loc_head = MLP(num_channels, hidden_channels + [1])
        self.class_head = MLP(num_channels, hidden_channels + [num_classes])
        self.quad_head = MLP(num_channels, hidden_channels + [8])

        self.output_shapes = {
            "num_instances": ("batch_size",),
            "scores": ("batch_size", max_instances),
            "classes": ("batch_size", max_instances),
            "quads": ("batch_size", max_instances, 4, 2),
        }

    def get_saliency(self, inputs: List[Tensor]) -> Tensor:
        device = inputs[self.bottom_level].device
        batch_size, _, full_height, full_width = inputs[self.bottom_level].shape
        output = torch.zeros((batch_size, full_height, full_width), device=device)
        global_context = rearrange(
            self.global_context(inputs[self.top_level]), "b c 1 1 -> b 1 c"
        )
        for lateral, level in zip(self.laterals, self.levels):
            height, width = inputs[level].shape[2:]
            feats = rearrange(lateral(inputs[level]), "b c h w -> b (h w) c")
            feats = feats + global_context
            scores = self.loc_head(feats).sigmoid()
            scores = rearrange(scores, "b (h w) c -> b c h w", h=height, w=width, c=1)
            scores = functional.interpolate(scores, size=(full_height, full_width))
            output = torch.maximum(output, scores.squeeze(1))
        return output

    def get_offsets_and_levels(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        device = inputs[0].device
        rel_offsets, levels = [], []
        for level in self.levels:
            h, w = inputs[level].shape[2:]
            y_min, x_min = 1 / h / 2, 1 / w / 2
            ys = torch.linspace(y_min, 1 - y_min, steps=h, device=device)
            xs = torch.linspace(x_min, 1 - x_min, steps=w, device=device)
            coordinate_grid = torch.stack(
                [repeat(xs, "w -> h w", h=h), repeat(ys, "h -> h w", w=w)], dim=2
            )
            rel_grid = rearrange(coordinate_grid, "h w c -> (h w) c")
            rel_offsets.append(rel_grid)
            levels.append(torch.full((h * w, 1), level, device=device))
        rel_offsets, levels = torch.cat(rel_offsets), torch.cat(levels)
        rel_offsets = repeat(rel_offsets, "i c -> i (4 c)", c=2)
        return rel_offsets, levels

    def get_features(self, inputs: List[Tensor]) -> Tensor:
        global_context = self.global_context(inputs[self.top_level])
        return [
            lateral(inputs[level]) + global_context
            for lateral, level in zip(self.laterals, self.levels)
        ]

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        device = inputs[0].device
        batch_size, _, full_height, full_width = inputs[0].shape
        feats = self.get_features(inputs)
        feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], dim=1)
        rel_offsets, levels = self.get_offsets_and_levels(inputs)
        # compute locations
        loc_logits = self.loc_head(feats).squeeze(2)
        loc_logits, loc_idxs = loc_logits.topk(self.max_instances, dim=1)
        batches = repeat(torch.arange(batch_size), f"b -> b {self.max_instances}")
        scores = loc_logits.sigmoid()
        num_instances = reduce(scores > 0.5, "b i -> b", "sum")
        feats = feats[batches, loc_idxs]
        # compute quads
        offsets = repeat(rel_offsets, "i c -> b i c", b=batch_size)[batches, loc_idxs]
        levels = repeat(levels, "i c -> b i c", b=batch_size)[batches, loc_idxs]
        quad_preds = (self.quad_head(feats)).tanh()
        quad_preds = offsets + quad_preds
        quad_preds = quad_preds * torch.tensor(
            [[[full_width, full_height] * 4]], device=device
        )
        quad_preds = rearrange(quad_preds, "b i (v d) -> b i v d", v=4, d=2)
        # compute classes
        class_logits = self.class_head(feats)
        classes = class_logits.max(dim=2).indices
        return num_instances, scores, classes, quad_preds

    def training_step(
        self,
        inputs: List[Tensor],
        classes: List[Tensor],
        quads: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert len(inputs) > self.top_level, "too few input levels"
        device = inputs[0].device
        batch_size, _, full_height, full_width = inputs[0].shape
        feats = self.get_features(inputs)
        feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], dim=1)

        # compute anchors
        rel_offsets, levels = self.get_offsets_and_levels(inputs)
        directions = torch.tensor([[-1, -1, 1, 1]], device=device)
        scale = torch.sigmoid(levels - self.top_level)
        anchors = (rel_offsets[:, :4] + directions * scale) * torch.tensor(
            [[full_width, full_height] * 2], device=device
        )
        # assign anchors to gt boxes
        matching_results = [
            self.bbox_matching(anchors, self.quads_to_boxes(q), self.topk)
            for q in quads
        ]
        assignment = torch.stack([_[0] for _ in matching_results])
        o2o_mask = torch.stack([_[1] for _ in matching_results])
        # o2m_iou = torch.stack([_[2] for _ in matching_results])
        rel_iou = torch.stack([_[3] for _ in matching_results])

        o2m_mask = rel_iou > 0
        loc_target = rel_iou / self.topk
        loc_target[o2o_mask] = 1
        o2m_weights = rel_iou[o2m_mask]
        o2m_feats = feats[o2m_mask]

        # compute quad loss
        offsets = torch.cat([rel_offsets[mask] for mask in o2m_mask])
        levels = torch.cat([levels[mask] for mask in o2m_mask])
        quad_preds = (self.quad_head(o2m_feats)).tanh()
        quad_preds = (offsets + quad_preds).clamp(0, 1)
        quad_preds = rearrange(quad_preds, "i (v d) -> i v d", v=4, d=2)

        quad_target = torch.cat(
            [quads[b][assignment[b, mask]] for b, mask in enumerate(o2m_mask)]
        )
        quad_target = self.canonicalize_and_convexify(quad_target) / torch.tensor(
            [[[full_width, full_height]] * 4], device=device
        )
        with torch.autocast(device_type="cuda", enabled=False):
            quad_loss = functional.l1_loss(quad_preds, quad_target, reduction="none")
            quad_loss = quad_loss.sum(dim=(1, 2))
            quad_loss = 10 * (o2m_weights * quad_loss).sum() / o2m_weights.sum()

        # compute classification loss
        class_logits = self.class_head(o2m_feats)
        class_target = torch.cat(
            [classes[b][assignment[b, mask]] for b, mask in enumerate(o2m_mask)]
        )
        class_target = functional.one_hot(class_target, self.num_classes)
        with torch.autocast(device_type="cuda", enabled=False):
            class_loss = ops.sigmoid_focal_loss(
                class_logits, class_target.to(torch.float32), reduction="none"
            ).sum(dim=1)
            class_loss = 10 * (o2m_weights * class_loss).sum() / o2m_weights.sum()

        # compute locations
        loc_logits = self.loc_head(feats).squeeze(2)
        with torch.autocast(device_type="cuda", enabled=False):
            loc_loss = functional.binary_cross_entropy_with_logits(
                loc_logits.to(torch.float32), loc_target, reduction="none"
            )
            loc_loss = loc_loss.sum() / loc_target.sum()

        loss = loc_loss + quad_loss + class_loss
        metrics = {
            "location_loss": loc_loss,
            "quad_loss": quad_loss,
            "class_loss": class_loss,
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
        self, inputs: List[Tensor], classes: List[Tensor], quads: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        num_instances, scores, pred_classes, quad_preds = self.forward(inputs)
        b, i, _, _ = quad_preds.shape
        box_preds = self.quads_to_boxes(
            rearrange(quad_preds, "b i v c -> (b i) v c", v=4, c=2)
        )
        box_preds = rearrange(box_preds, "(b i) v -> b i v", b=b, i=i, v=4)
        boxes = [self.quads_to_boxes(_) for _ in quads]
        self.map_computer.to(scores.device).update(
            [
                {"scores": s, "labels": c, "boxes": b}
                for s, c, b in zip(scores, pred_classes, box_preds)
            ],
            [{"labels": c, "boxes": b} for c, b in zip(classes, boxes)],
        )
        loss, metrics = self.training_step(inputs, classes, quads, is_validating=True)
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
        anchors: Tensor, gt_boxes: Tensor, topk: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        device = anchors.device
        num_anchors, num_gt = anchors.shape[0], gt_boxes.shape[0]
        o2m_assignments = torch.full((num_anchors,), -1, device=device)
        o2o_mask = torch.zeros((num_anchors,), dtype=torch.bool, device=device)
        o2m_iou = torch.zeros((num_anchors,), device=device)
        o2m_rel_iou = torch.zeros((num_anchors,), device=device)
        if num_gt == 0:
            return o2m_assignments, o2o_mask, o2m_iou, o2m_rel_iou
        ious = ops.complete_box_iou(anchors, gt_boxes)
        topk_ious, topk_idxs = torch.topk(ious, k=topk, dim=0)
        is_best_match = torch.zeros((num_anchors, num_gt), dtype=bool, device=device)
        is_best_match.scatter_(0, topk_idxs[0:1], True)
        is_topk_match = torch.zeros((num_anchors, num_gt), dtype=bool, device=device)
        is_topk_match.scatter_(0, topk_idxs, True)
        max_ious, max_gt_idxs = torch.max(ious * is_topk_match.float(), dim=1)
        valid_mask = is_topk_match.any(dim=1)
        o2m_assignments[valid_mask] = max_gt_idxs[valid_mask]
        o2o_mask = is_best_match.any(dim=1)
        o2m_iou[valid_mask] = max_ious[valid_mask]
        # compute relative iou
        best_ious_per_gt = topk_ious[0]
        best_iou_for_assignment = best_ious_per_gt[max_gt_idxs]
        o2m_rel_iou[valid_mask] = (
            max_ious[valid_mask] / best_iou_for_assignment[valid_mask]
        ).nan_to_num(0)
        return o2m_assignments, o2o_mask, o2m_iou, o2m_rel_iou

    @staticmethod
    def canonicalize_and_convexify(quads: Tensor) -> Tensor:
        # sort vertices by angle around centroid
        centroid = reduce(quads, "n v c -> n 1 c", "mean", v=4, c=2)
        rel = quads - centroid
        angles = torch.atan2(rel[..., 1], rel[..., 0])
        _, order = angles.sort(dim=1)
        idx = repeat(order, "n v -> n v c", v=4, c=2)
        v = torch.gather(quads, 1, idx)
        v_next = v[:, [1, 2, 3, 0], :]
        v_prev = v[:, [3, 0, 1, 2], :]
        # concavity detection via cross-product
        cross = (v_next[..., 0] - v[..., 0]) * (v_prev[..., 1] - v[..., 1]) - (
            (v_next[..., 1] - v[..., 1]) * (v_prev[..., 0] - v[..., 0])
        )
        concave_mask = cross < 0
        # replace concave vertices with neighbor midpoint
        mid = (v_prev + v_next) * 0.5
        mask = rearrange(concave_mask, "n v -> n v 1", v=4)
        v_fixed = torch.where(mask, mid, v)
        return v_fixed

    @staticmethod
    def quads_to_boxes(quads: Tensor) -> Tensor:
        x, y = quads[..., 0], quads[..., 1]
        boxes = torch.stack(
            [x.min(-1).values, y.min(-1).values, x.max(-1).values, y.max(-1).values], 1
        )
        return boxes

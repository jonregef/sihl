from functools import partial
from typing import Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchvision import ops
import torch

from sihl.utils import coordinate_grid, polygon_iou
from sihl.heads.object_detection import ObjectDetection


class QuadrilateralDetection(ObjectDetection):
    """Quadrilateral detection is like object detection, expect objects are associated
    with convex quadrilaterals instead of axis-aligned rectangles."""

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
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            bottom_level=bottom_level,
            top_level=top_level,
            num_channels=num_channels,
            num_layers=num_layers,
            max_instances=max_instances,
        )

        MLP = partial(ops.MLP, norm_layer=nn.LayerNorm, activation_layer=nn.SiLU)
        self.quad_offset_head = MLP(num_channels, [num_channels] * num_layers + [4])

        self.output_shapes = {
            "num_instances": ("batch_size",),
            "scores": ("batch_size", self.max_instances),
            "classes": ("batch_size", self.max_instances),
            "quadrilaterals": ("batch_size", self.max_instances, 4, 2),
        }

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        device = inputs[0].device
        batch_size, _, full_height, full_width = inputs[0].shape
        feats = self.get_features(inputs)
        feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], dim=1)
        _, _, abs_offsets, abs_scales = self.get_offsets_and_scales(inputs)
        # compute locations
        loc_logits = self.loc_head(feats).squeeze(2)
        loc_logits, loc_idxs = loc_logits.topk(self.max_instances, dim=1)
        batches = repeat(torch.arange(batch_size), f"b -> b {self.max_instances}")
        scores = loc_logits.sigmoid()
        num_instances = reduce(scores > 0.5, "b i -> b", "sum")
        feats = feats[batches, loc_idxs]
        # compute boxes
        offsets = repeat(abs_offsets, "i c -> b i c", b=batch_size)[batches, loc_idxs]
        scales = repeat(abs_scales, "i c -> b i c", b=batch_size)[batches, loc_idxs]
        directions = torch.tensor([[[-1, -1, 1, 1]]], device=device)
        box_preds = self.box_head(feats).exp()
        box_preds = offsets + directions * scales * box_preds
        # compute classes
        class_logits = self.class_head(feats)
        classes = class_logits.max(dim=2).indices

        xmin, ymin, xmax, ymax = box_preds.unbind(dim=2)
        offsets = self.quad_offset_head(feats).sigmoid()
        quad_preds = torch.stack(
            [
                torch.stack([xmin, ymin + offsets[:, :, 0] * (ymax - ymin)], dim=2),
                torch.stack([xmax - offsets[:, :, 1] * (xmax - xmin), ymin], dim=2),
                torch.stack([xmax, ymax - offsets[:, :, 2] * (ymax - ymin)], dim=2),
                torch.stack([xmin + offsets[:, :, 3] * (xmax - xmin), ymax], dim=2),
            ],
            dim=2,
        )
        return num_instances, scores, classes, quad_preds

    def training_step(
        self,
        inputs: List[Tensor],
        classes: List[Tensor],
        quads: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert len(inputs) > self.top_level, "too few input levels"

        boxes = [quads_to_boxes(_) for _ in quads]
        device = inputs[0].device
        batch_size, _, full_height, full_width = inputs[0].shape
        feats = self.get_features(inputs)
        feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], dim=1)

        # compute anchors
        _, _, abs_offsets, abs_scales = self.get_offsets_and_scales(inputs)
        directions = torch.tensor([[-1, -1, 1, 1]], device=device)
        anchors = abs_offsets + directions * abs_scales

        # assign pred boxes to gt boxes
        matching_results = [
            self.bbox_matching(anchors, boxes[b]) for b in range(batch_size)
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

        # compute box loss
        box_offsets = torch.cat([abs_offsets[mask] for mask in o2m_mask])
        box_scales = torch.cat([abs_scales[mask] for mask in o2m_mask])
        box_preds = self.box_head(o2m_feats).exp()
        box_preds = box_offsets + directions * box_scales * box_preds

        box_target = torch.cat(
            [boxes[b][assignment[b, mask]] for b, mask in enumerate(o2m_mask)]
        )
        with torch.autocast(device_type="cuda", enabled=False):
            box_loss = ops.complete_box_iou_loss(
                box_preds.to(torch.float32), box_target, reduction="none"
            )
            box_loss = 10 * (o2m_weights * box_loss).sum() / o2m_weights.sum()

        # compute classification loss
        class_logits = self.class_head(o2m_feats)
        class_target = torch.cat(
            [classes[b][assignment[b, mask]] for b, mask in enumerate(o2m_mask)]
        )
        class_target = functional.one_hot(class_target, self.num_classes)
        with torch.autocast(device_type="cuda", enabled=False):
            class_loss = ops.sigmoid_focal_loss(
                class_logits, class_target.to(torch.float32), reduction="none"
            )
            o2m_weights = o2m_weights.unsqueeze(1)
            class_loss = 10 * (o2m_weights * class_loss).sum() / o2m_weights.sum()

        # compute locations
        loc_logits = self.loc_head(feats).squeeze(2)
        with torch.autocast(device_type="cuda", enabled=False):
            loc_loss = functional.binary_cross_entropy_with_logits(
                loc_logits.to(torch.float32), loc_target, reduction="none"
            )
            loc_loss = loc_loss.sum().relu() / loc_target.sum()

        # compute quad loss
        xmin, ymin, xmax, ymax = box_preds.detach().unbind(dim=1)
        offsets = self.quad_offset_head(o2m_feats.detach()).sigmoid()
        quad_preds = torch.stack(
            [
                torch.stack([xmin, ymin + offsets[:, 0] * (ymax - ymin)], dim=1),
                torch.stack([xmin + offsets[:, 1] * (xmax - xmin), ymin], dim=1),
                torch.stack([xmax, ymin + offsets[:, 2] * (ymax - ymin)], dim=1),
                torch.stack([xmin + offsets[:, 3] * (xmax - xmin), ymax], dim=1),
            ],
            dim=1,
        )
        quad_target = torch.cat(
            [quads[b][assignment[b, mask]] for b, mask in enumerate(o2m_mask)]
        )

        with torch.autocast(device_type="cuda", enabled=False):
            """https://arxiv.org/abs/2103.11636"""
            permutations = []
            for shift in range(4):
                shifted_quad = torch.roll(quad_target, shifts=-shift, dims=1)
                permutations.append(shifted_quad)
                permutations.append(shifted_quad.flip(dims=(1,)))
            permutations = torch.stack(permutations, dim=1)  # (N, 8, 4, 2)
            quad_preds = quad_preds.unsqueeze(1).repeat_interleave(8, dim=1)
            full_size = torch.tensor([[[[full_width, full_height]]]], device=device)
            quad_target = permutations / full_size
            quad_preds = quad_preds / full_size
            quad_loss = functional.smooth_l1_loss(
                quad_preds, quad_target, reduction="none"
            )
            # average loss over quad vertices and space, take minimum of permutations
            quad_loss = quad_loss.mean((1, 2)).min(1).values
            quad_loss = (o2m_weights * quad_loss).sum() / o2m_weights.sum()

        loss = loc_loss + box_loss + class_loss + quad_loss
        metrics = {
            "location_loss": loc_loss,
            "box_loss": box_loss,
            "class_loss": class_loss,
            "quad_loss": quad_loss,
        }
        return loss, metrics

    def validation_step(
        self, inputs: List[Tensor], classes: List[Tensor], quads: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        num_instances, scores, pred_classes, pred_quads = self.forward(inputs)
        self.map_computer.to(pred_quads.device).update(
            [
                {"scores": s, "labels": c, "boxes": b}
                for s, c, b in zip(scores, pred_classes, quads_to_boxes(pred_quads))
            ],
            [{"labels": c, "boxes": quads_to_boxes(q)} for c, q in zip(classes, quads)],
        )
        loss, metrics = self.training_step(inputs, classes, quads, is_validating=True)
        self.loss_computer.to(loss.device).update(loss)
        return loss, metrics


def quads_to_boxes(quads):
    x, y = quads[..., 0], quads[..., 1]
    return torch.stack([x.min(-1)[0], y.min(-1)[0], x.max(-1)[0], y.max(-1)[0]], dim=-1)


def line_equation(p1: Tensor, p2: Tensor) -> Tuple[float, float, float]:
    """find (a, b, c) such that ax + by + c = 0"""
    a, b, c = p2[1] - p1[1], p1[0] - p2[0], p2[0] * p1[1] - p1[0] * p2[1]
    return a, b, c


def inside_polygon(points: Tensor, polygon: Tensor) -> Tensor:
    """A point is inside a polygon if it is inside all edges' half-planes.
    N points (N, 2), 1 M-sided polygon (M, 2) -> (N,) bool
    """
    m = polygon.shape[0]
    edges = [line_equation(polygon[i], polygon[(i + 1) % m]) for i in range(m)]
    inside = [a * points[:, 0] + b * points[:, 1] + c >= 0 for (a, b, c) in edges]
    return torch.stack(inside).all(dim=0)


def uncross(quadrilaterals: Tensor) -> Tensor:
    """Take arbitrary quadrilaterals and reorder their vertices according to their angle
    to the topmost vertex.

    Args:
        quadrilaterals (Tensor[N, 4, 2]): A batch of quadrilaterals.

    Returns:
        Tensor[N, 4, 2]: The same batch of quadrilaterals, with vertices re-ordered.
    """
    n = quadrilaterals.shape[0]
    topmost_idxs = torch.argmin(quadrilaterals[:, :, 1], dim=1)
    topmost_points = quadrilaterals[torch.arange(n), topmost_idxs].unsqueeze(1)
    shifted = quadrilaterals - topmost_points
    angles = torch.atan2(shifted[:, :, 1], shifted[:, :, 0])
    angles = torch.where(angles < 0, angles + 2 * torch.pi, angles)  # wrap angles
    # set the angle of the topmost vertex to -1 to ensure it's first after sorting
    angles[torch.arange(n), topmost_idxs] = -1  # FIXME: do not mutate
    _, sorted_indices = torch.sort(angles, dim=1)
    reordered = torch.gather(
        quadrilaterals, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 2)
    )
    return reordered


def convexify(quadrilaterals: Tensor) -> Tensor:
    """Take arbitrary quadrilaterals and convexify them.

    Args:
        quadrilaterals (Tensor[N, 4, 2]): A batch of quadrilaterals.

    Returns:
        Tensor[N, 4, 2]: The same batch of quadrilaterals, with vertices re-ordered
        and adjusted for concave quadrilaterals.
    """
    uncrossed = uncross(quadrilaterals)
    edges = torch.roll(uncrossed, -1, dims=1) - uncrossed
    cross_products = (
        edges[:, :, 0] * torch.roll(edges, -1, dims=1)[:, :, 1]
        - edges[:, :, 1] * torch.roll(edges, -1, dims=1)[:, :, 0]
    )
    # For concave quads, replace the point opposite (idx 2) to the pivot (idx 0)
    # with the center of the segment bounded by the two other points (idxs 1 & 3)
    is_concave = torch.any(cross_products < 0, dim=1).reshape(-1, 1, 1)
    center_points = is_concave * ((uncrossed[:, 1] + uncrossed[:, 3]) / 2).unsqueeze(1)
    zeros = torch.zeros_like(center_points)
    center_points = torch.cat([zeros, zeros, center_points, zeros], dim=1)
    # replace c with c': [a,b,c',d] = [a,b,c,d] * [1,1,0,1] + [0,0,c',0]
    uncrossed = uncrossed * (~center_points.to(torch.bool)) + center_points
    return uncrossed

from typing import Any, Tuple, List, Dict

from einops import rearrange, reduce
from torch import Tensor
from torch.nn import functional
import torch

from sihl.utils import EPS

from .instance_segmentation import InstanceSegmentation


class KeypointDetection(InstanceSegmentation):
    """Keypoint detection is like object detection, except instances are all of the same
    category (i.e. `num_classes` = 1), and are associated with a set of 2D (key)points
    instead of an axis-aligned bounding box. Each instance predictions must provide a 2D
    coordinate for every keypoint, as well as whether each keypoint is present in the
    image or not (i.e. missing or invisible). Coordinate predictions of non-present
    keypoints are safe to ignore.

    Refs:
        1. [FCPose](https://arxiv.org/abs/2105.14185)
    """

    def __init__(
        self,
        in_channels: List[int],
        num_keypoints: int,
        bottom_level: int = 3,
        top_level: int = 7,
        num_channels: int = 256,
        num_layers: int = 4,
        max_instances: int = 32,
        t_min: float = 0.2,
        t_max: float = 0.6,
        topk: int = 7,
        soft_label_decay_steps: int = 1,
        mask_top_level: int = 5,
        mask_num_channels: int = 32,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_keypoints (int): Number of keypoints.
            bottom_level (int, optional): Bottom level of inputs this head is attached to. Defaults to 3.
            top_level (int, optional): Top level of inputs this head is attached to. Defaults to 7.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 4.
            max_instances (int, optional): Maximum number of instances to predict in a sample. Defaults to 32.
            t_min (float, optional): Lower bound of O2F parameter t. Defaults to 0.2.
            t_max (float, optional): Upper bound of O2F parameter t. Defaults to 0.6.
            topk (int, optional): How many anchors to match to each ground truth object when copmuting the loss. Defaults to 7.
            soft_label_decay_steps (int, optional): How many training steps to perform before the one-to-few matching becomes one-to-one. Defaults to 1.
            mask_top_level (int, optional): Top level of inputs masks are computed from. Defaults to 5.
            mask_num_channels (int, optional): Number of convolutional channels in the keypoint heatmap prediction branch.
        """
        super().__init__(
            num_classes=1,
            in_channels=in_channels,
            num_channels=num_channels,
            num_layers=num_layers,
            bottom_level=bottom_level,
            top_level=top_level,
            max_instances=max_instances,
            t_min=t_min,
            t_max=t_max,
            topk=topk,
            soft_label_decay_steps=soft_label_decay_steps,
            mask_top_level=mask_top_level,
            mask_num_channels=mask_num_channels,
            mask_out_channels=num_keypoints,
        )
        self.num_kpts = num_keypoints
        self.output_shapes = {
            "num_instances": ("batch_size",),
            "instance_scores": ("batch_size", max_instances),
            "keypoint_scores": ("batch_size", max_instances, num_keypoints),
            "keypoints": ("batch_size", max_instances, num_keypoints, 2),
        }

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_instances, scores, _, logits = self.forward_logits(inputs)
        b, _, h, w = logits.shape
        heatmaps = rearrange(
            logits, "b (i k) h w -> b i k h w", i=self.max_instances, k=self.num_kpts
        )
        heatmaps = heatmaps.flatten(3, 4).softmax(dim=3).reshape(heatmaps.shape)
        keypoint_scores = reduce(heatmaps, "b i k h w -> b i k", "max")
        keypoints = self.heatmap_to_keypoints(inputs, heatmaps)
        return num_instances, scores, keypoint_scores, keypoints

    def heatmap_to_keypoints(self, inputs: List[Tensor], heatmaps: Tensor) -> Tensor:
        heatmap_height, heatmap_width = heatmaps.shape[3:]
        kpt_pos = rearrange(heatmaps, "b i k h w -> b i k (h w)").argmax(dim=3)
        img_height, img_width = inputs[0].shape[2:]
        y = ((kpt_pos // heatmap_width) / heatmap_height) * img_height
        x = ((kpt_pos % heatmap_width) / heatmap_width) * img_width
        return torch.stack([x, y], dim=-1)

    def training_step(
        self,
        inputs: List[Tensor],
        keypoints: List[Tensor],
        presence: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        batch_size, _, img_height, img_width = inputs[0].shape
        scores, reg_features = self.get_features(inputs)
        controllers = self.get_controllers(reg_features)
        pred_boxes = self.get_boxes(reg_features)
        bottom_features = self.get_bottom_features(inputs)
        anchors = self.get_anchors(inputs)  # (2, H, W)
        normalized_anchors = self.get_anchors(inputs, normalized=True)

        boxes = [keypoints_to_boxes(k, p) for (k, p) in zip(keypoints, presence)]
        classes = [
            torch.zeros((k.shape[0],), dtype=torch.int64, device=scores.device)
            for k in keypoints
        ]

        class_target, box_target, assignment = self.get_targets(
            anchors, scores, pred_boxes, classes, boxes, is_validating
        )
        class_loss, box_loss = self.get_losses(
            scores, pred_boxes, class_target, box_target
        )

        heatmap_losses, distance_losses = [], []
        heatmap_height, heatmap_width = bottom_features.shape[2:]
        for batch_idx in range(batch_size):
            pos_idxs = class_target[batch_idx].max(dim=1).values >= 1 - EPS
            if keypoints[batch_idx].numel() == 0 or not torch.any(pos_idxs):
                continue

            anchor_idxs = assignment[batch_idx, pos_idxs]
            target_coords = keypoints[batch_idx][anchor_idxs].clone()
            target_coords[:, :, 0] *= (heatmap_width - 1) / (img_width - 1)
            target_coords[:, :, 1] *= (heatmap_height - 1) / (img_height - 1)
            target_coords = target_coords.round().to(torch.int64)
            gt_xs = functional.one_hot(target_coords[:, :, 0], heatmap_width)
            gt_ys = functional.one_hot(target_coords[:, :, 1], heatmap_height)

            target_heatmap = (gt_xs.unsqueeze(2) * gt_ys.unsqueeze(3)).to(class_loss)
            pred_heatmap = self.get_masks(
                params=controllers[batch_idx][pos_idxs],
                anchors=normalized_anchors[pos_idxs],
                bottom_features=bottom_features[batch_idx],
                num_masks=pos_idxs.sum(),
            ).reshape((-1, self.num_kpts, heatmap_height, heatmap_width))

            presence_mask = presence[batch_idx][anchor_idxs].flatten()
            shape = (-1, heatmap_height * heatmap_width)  # CE over flattened space
            flat_pred_hm = pred_heatmap.reshape(shape)[presence_mask]
            flat_target_hm = target_heatmap.reshape(shape)[presence_mask]
            with torch.autocast(device_type="cuda", enabled=False):
                ce_loss = functional.cross_entropy(
                    flat_pred_hm.to(torch.float32), flat_target_hm, reduction="none"
                )
                heatmap_losses.append(ce_loss.nan_to_num(0).mean())

            peaks = flat_pred_hm.argmax(1)
            peak_rows, peaks_cols = peaks // heatmap_width, peaks % heatmap_width
            pred_keypoints = torch.stack(
                [peaks_cols / (heatmap_width - 1), peak_rows / (heatmap_height - 1)],
                dim=1,
            )
            gt_keypoints = target_coords.flatten(0, 1)[presence_mask].to(pred_keypoints)
            gt_keypoints[:, 0] /= heatmap_width - 1
            gt_keypoints[:, 1] /= heatmap_height - 1
            distance_losses.append((gt_keypoints - pred_keypoints).pow(2).mean())

        heatmap_loss = torch.stack(heatmap_losses).mean() if heatmap_losses else 0
        distance_loss = torch.stack(distance_losses).mean() if distance_losses else 0

        return class_loss + box_loss + heatmap_loss + distance_loss, {
            "class_loss": class_loss,
            "box_loss": box_loss,
            "heatmap_loss": heatmap_loss,
            "distance_loss": distance_loss,
        }

    def validation_step(
        self,
        inputs: List[Tensor],
        keypoints: List[Tensor],
        presence: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        boxes = [keypoints_to_boxes(*args) for args in zip(keypoints, presence)]
        _, scores, kpt_scores, pred_keypoints = self.forward(inputs)
        pred_boxes = [
            keypoints_to_boxes(kpts, pres > 0.1)
            for (kpts, pres) in zip(pred_keypoints, kpt_scores)
        ]
        dev = scores.device
        self.map_computer.to(dev).update(
            [
                {"scores": s, "labels": torch.zeros_like(s, dtype=int), "boxes": b}
                for s, b in zip(scores, pred_boxes)
            ],
            [
                {"labels": torch.zeros(b.shape[0], dtype=int, device=dev), "boxes": b}
                for b in boxes
            ],
        )
        loss, metrics = self.training_step(inputs, keypoints, presence, True)
        self.loss_computer.to(dev).update(loss)
        return loss, metrics


def keypoints_to_boxes(keypoints: Tensor, presence: Tensor) -> Tensor:
    assert presence.dtype == torch.bool
    masked_keypoints = keypoints.clone()
    masked_keypoints[~presence] = torch.inf
    xmin = masked_keypoints[:, :, 0].min(dim=1).values
    ymin = masked_keypoints[:, :, 1].min(dim=1).values
    masked_keypoints[~presence] = -torch.inf
    xmax = masked_keypoints[:, :, 0].max(dim=1).values
    ymax = masked_keypoints[:, :, 1].max(dim=1).values
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


# FIXME
# def compute_oks(
#     pred_keypoints: Tensor, gt_keypoints: Tensor, presence: Tensor
# ) -> Tensor:
#     dists = torch.sum((pred_keypoints - gt_keypoints) ** 2, dim=2)  # Shape: (B, K)
#     bboxes = keypoints_to_boxes(gt_keypoints, presence)
#     areas = ((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])).unsqueeze(1)
#     sigmas = torch.full((gt_keypoints.shape[1],), 0.05, device=pred_keypoints.device)
#     exp_terms = torch.exp(-dists / (sigmas**2) * 2 * (areas**2)) * presence  # (B, K)
#     oks_scores = exp_terms.sum(dim=1) / presence.sum(dim=1)
#     return oks_scores

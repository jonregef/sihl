from functools import partial
from typing import Any, Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import ops
import torch

from sihl.utils import coordinate_grid
from sihl.heads.object_detection import ObjectDetection


class InstanceSegmentation(ObjectDetection):
    """Instance segmentation is the prediction of the set of "objects" (pairs of binary
    mask and the corresponding category) in the input image. You can see it as object
    detection but instead of bounding boxes we predict binary masks.

    Refs:
        1. [CondInst](https://arxiv.org/abs/2102.03026)
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
            top_level (int, optional): Top level of inputs this head is attached to. Defaults to 5.
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
        Conv = partial(ops.Conv2dNormActivation, activation_layer=nn.SiLU)

        c = self.mask_num_channels = 8
        kernel_params = (c + 2) * c + c + c * c + c + c * 1 + 1
        self.kernel_head = MLP(
            num_channels, [num_channels for _ in range(num_layers)] + [kernel_params]
        )
        self.mask_laterals = nn.ModuleList(
            [Conv(in_channels[lvl], num_channels, 1) for lvl in self.levels]
        )
        self.mask_stem = Conv(num_channels, self.mask_num_channels, 3)
        scale = 2**bottom_level
        self.output_shapes = {
            "num_instances": ("batch_size",),
            "scores": ("batch_size", max_instances),
            "classes": ("batch_size", max_instances),
            "masks": ("batch_size", max_instances, f"height/{scale}", f"width/{scale}"),
        }

    def forward(
        self, inputs: List[Tensor], faster: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        device = inputs[0].device
        batch_size, full_size = inputs[0].shape[0], inputs[0].shape[2:]
        feats = self.get_features(inputs)
        flat_feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], 1)

        _, _, abs_offsets, abs_scales = self.get_offsets_and_scales(inputs)

        # compute locations
        loc_logits = self.loc_head(flat_feats).squeeze(2)
        loc_logits, loc_idxs = loc_logits.topk(self.max_instances, dim=1)
        scores = loc_logits.sigmoid()
        num_instances = reduce(scores > 0.5, "b i -> b", "sum")
        batches = repeat(torch.arange(batch_size), f"b -> b {self.max_instances}")
        flat_feats = flat_feats[batches, loc_idxs]

        # compute classes
        class_logits = self.class_head(flat_feats)
        classes = class_logits.max(dim=2).indices

        # compute masks
        mask_height, mask_width = inputs[self.bottom_level].shape[2:]
        mask_feats = [
            functional.interpolate(x, (mask_height, mask_width), mode="bilinear")
            for x in feats
        ]
        mask_feats = self.mask_stem(torch.stack(mask_feats).sum(dim=0))
        mask_feats = repeat(mask_feats, "b c h w -> b i c h w", i=self.max_instances)

        rel_offsets = [
            coordinate_grid(*inputs[lvl].shape[2:]).to(device) for lvl in self.levels
        ]
        rel_offsets = torch.cat(
            [repeat(_, "h w c -> b (h w) c", b=batch_size) for _ in rel_offsets], dim=1
        )
        rel_offsets = rel_offsets[batches, loc_idxs]
        grid = coordinate_grid(mask_height, mask_width).to(device)
        grid = repeat(grid, "h w c -> b i c h w", b=batch_size, i=self.max_instances)
        grid = grid - rearrange(rel_offsets, "b i c -> b i c 1 1", i=self.max_instances)
        mask_feats = torch.cat([mask_feats, grid], dim=2)

        if faster:
            # NOTE: this vectorized implementation is unfortunately not onnx exportable
            mask_feats = rearrange(mask_feats, "b i c h w -> 1 (b i c) h w")
            w = self.kernel_head(flat_feats)
            c = self.mask_num_channels
            n_groups = batch_size * self.max_instances
            bic = n_groups * self.mask_num_channels
            w1 = w[..., (s := slice(0, (c + 2) * c))].reshape(bic, c + 2, 1, 1)
            b1 = w[..., (s := slice(s.stop, s.stop + c))].reshape(bic)
            w2 = w[..., (s := slice(s.stop, s.stop + c**2))].reshape(bic, c, 1, 1)
            b2 = w[..., (s := slice(s.stop, s.stop + c))].reshape(bic)
            w3 = w[..., (s := slice(s.stop, s.stop + c))].reshape(n_groups, c, 1, 1)
            b3 = w[..., s.stop :].reshape(n_groups)
            masks = functional.silu(torch.conv2d(mask_feats, w1, b1, groups=n_groups))
            masks = functional.silu(torch.conv2d(masks, w2, b2, groups=n_groups))
            masks = functional.sigmoid(torch.conv2d(masks, w3, b3, groups=n_groups))
            masks = rearrange(
                masks, "1 (b i) h w -> b i h w", b=batch_size, i=self.max_instances
            )
        else:
            mask_feats = rearrange(mask_feats, "b i c h w -> b (i c) h w")
            masks_list = []
            c = self.mask_num_channels
            n_groups = self.max_instances
            bic = n_groups * self.mask_num_channels
            dynamic_weights = self.kernel_head(flat_feats)
            for b in range(batch_size):
                w = dynamic_weights[b]
                w1 = w[..., (s := slice(0, (c + 2) * c))].reshape(bic, c + 2, 1, 1)
                b1 = w[..., (s := slice(s.stop, s.stop + c))].reshape(bic)
                w2 = w[..., (s := slice(s.stop, s.stop + c**2))].reshape(bic, c, 1, 1)
                b2 = w[..., (s := slice(s.stop, s.stop + c))].reshape(bic)
                w3 = w[..., (s := slice(s.stop, s.stop + c))].reshape(n_groups, c, 1, 1)
                b3 = w[..., s.stop :].reshape(n_groups)
                masks = functional.silu(
                    torch.conv2d(mask_feats[b : b + 1], w1, b1, groups=n_groups)
                )
                masks = functional.silu(torch.conv2d(masks, w2, b2, groups=n_groups))
                masks = torch.conv2d(masks, w3, b3, groups=n_groups)
                masks_list.append(masks[0])
            masks = torch.stack(masks_list).sigmoid()

        masks = functional.interpolate(masks, full_size, mode="bilinear")
        return num_instances, scores, classes, masks

    def training_step(
        self,
        inputs: List[Tensor],
        classes: List[Tensor],
        masks: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        assert len(inputs) > self.top_level, "too few input levels"
        device = inputs[0].device
        batch_size, _, full_height, full_width = inputs[0].shape

        # remove empty masks
        classes = [
            _classes[_masks.any(dim=(1, 2))] if _masks.shape[0] > 0 else _classes
            for _classes, _masks in zip(classes, masks)
        ]
        masks = [
            _masks[_masks.any(dim=(1, 2))] if _masks.shape[0] > 0 else _masks
            for _masks in masks
        ]
        boxes = [ops.masks_to_boxes(mask) for mask in masks]
        loss, metrics = super().training_step(inputs, classes, boxes, is_validating)

        # compute anchors
        _, _, abs_offsets, abs_scales = self.get_offsets_and_scales(inputs)
        directions = torch.tensor([[-1, -1, 1, 1]], device=device)
        anchors = abs_offsets + directions * abs_scales

        # assign anchors to gt boxes
        matching_results = [self.bbox_matching(anchors, gt_boxes) for gt_boxes in boxes]
        assignment = torch.stack([_[0] for _ in matching_results])
        # o2o_mask = torch.stack([_[1] for _ in matching_results])
        # o2m_iou = torch.stack([_[2] for _ in matching_results])
        rel_iou = torch.stack([_[3] for _ in matching_results])
        o2m_mask = rel_iou > 0
        o2m_weight = rel_iou[o2m_mask]

        feats = self.get_features(inputs)
        flat_feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], 1)

        mask_size = inputs[self.bottom_level].shape[2:]
        mask_feats = [
            functional.interpolate(x, mask_size, mode="bilinear") for x in feats
        ]
        mask_feats = self.mask_stem(torch.stack(mask_feats).sum(dim=0))
        rel_offsets = [
            coordinate_grid(*inputs[lvl].shape[2:]).to(device) for lvl in self.levels
        ]
        rel_offsets = torch.cat([rearrange(_, "h w c -> (h w) c") for _ in rel_offsets])
        grid = coordinate_grid(mask_size[0], mask_size[1]).to(device)
        biased_mask_feats = []
        for batch_idx in range(batch_size):
            loc_idxs = o2m_mask[batch_idx].nonzero()[:, 0]
            n_objects = loc_idxs.shape[0]
            if n_objects == 0:
                continue
            _mask_feats = repeat(mask_feats[batch_idx], "c h w -> i c h w", i=n_objects)
            _rel_offsets = rearrange(rel_offsets[loc_idxs], "i c -> i c 1 1")
            _grid = repeat(grid, "h w c ->  i c h w", i=n_objects) - _rel_offsets
            _mask_feats = torch.cat([_mask_feats, _grid], dim=1)
            _mask_feats = rearrange(_mask_feats, "i c h w -> (i c) h w")
            biased_mask_feats.append(_mask_feats)

        biased_mask_feats = torch.cat(biased_mask_feats).unsqueeze(0)
        dyn_weights = self.kernel_head(flat_feats[o2m_mask])
        n_obj = dyn_weights.shape[0]
        c = self.mask_num_channels
        bic = n_obj * c
        w1 = dyn_weights[:, (s := slice(0, (c + 2) * c))].reshape(bic, c + 2, 1, 1)
        b1 = dyn_weights[:, (s := slice(s.stop, s.stop + c))].reshape(bic)
        w2 = dyn_weights[:, (s := slice(s.stop, s.stop + c**2))].reshape(bic, c, 1, 1)
        b2 = dyn_weights[:, (s := slice(s.stop, s.stop + c))].reshape(bic)
        w3 = dyn_weights[:, (s := slice(s.stop, s.stop + c))].reshape(n_obj, c, 1, 1)
        b3 = dyn_weights[:, s.stop :].reshape(n_obj)
        masks_preds = functional.silu(
            torch.conv2d(biased_mask_feats, w1, b1, groups=n_obj)
        )
        masks_preds = functional.silu(torch.conv2d(masks_preds, w2, b2, groups=n_obj))
        masks_preds = torch.sigmoid(torch.conv2d(masks_preds, w3, b3, groups=n_obj))
        masks_preds = masks_preds.squeeze(0)
        target_masks = torch.cat(
            [masks[b][assignment[b, o2m_mask[b]]] for b in range(batch_size)]
        )
        target_masks = functional.interpolate(
            target_masks[:, None].to(masks_preds), mask_size, mode="bilinear"
        )[:, 0]
        # dice loss
        numerator = reduce(masks_preds * target_masks, "c h w -> c", "sum")
        denominator = reduce(masks_preds**2 + target_masks**2, "c h w -> c", "sum")
        with torch.autocast(device_type="cuda", enabled=False):
            mask_loss = 1 - 2 * numerator.to(torch.float32) / denominator
            mask_loss = (o2m_weight * mask_loss.nan_to_num(0)).sum() / o2m_weight.sum()
        loss = loss + mask_loss
        metrics["mask_loss"] = mask_loss
        return loss, metrics

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        max_detection_thresholds = [1, min(self.max_instances, 10), self.max_instances]
        self.map_computer = MeanAveragePrecision(
            max_detection_thresholds=max_detection_thresholds,
            backend="faster_coco_eval",
            iou_type="segm",
        )

    def validation_step(
        self, inputs: List[Tensor], classes: List[Tensor], masks: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        _, scores, pred_classes, pred_masks = self.forward(inputs, faster=True)
        self.map_computer.to(scores.device).update(
            [
                {"scores": s, "labels": c, "masks": m}
                for s, c, m in zip(scores, pred_classes, pred_masks > 0.5)
            ],
            [{"labels": c, "masks": m > 0.5} for c, m in zip(classes, masks)],
        )
        loss, metrics = self.training_step(inputs, classes, masks, is_validating=True)
        self.loss_computer.to(loss.device).update(loss)
        return loss, metrics

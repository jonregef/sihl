from functools import partial
from typing import Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import ops
import torch

from sihl.heads.object_detection import ObjectDetection


class InstanceSegmentation(nn.Module):
    """
    Refs:
        1. [Conditional Convolutions for Instance Segmentation](https://arxiv.org/abs/2003.05664)
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        mask_level: int = 3,
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
            mask_level (int, optional): Feature level of inputs used to compute keypoint heatmaps. Defaults to 3.
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
        self.mask_level = mask_level
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

        c = self.mask_num_channels = 8
        kernel_params = (c + 2) * c + c + c * c + c + c * 1 + 1
        self.kernel_head = MLP(num_channels, hidden_channels + [kernel_params])
        self.mask_lateral = Conv(in_channels[self.mask_level], num_channels, 1)
        self.mask_head = ops.Conv2dNormActivation(
            num_channels, c, 3, activation_layer=nn.SiLU
        )

        scale = 2**bottom_level
        self.output_shapes = {
            "num_instances": ("batch_size",),
            "scores": ("batch_size", max_instances),
            "classes": ("batch_size", max_instances),
            "masks": ("batch_size", max_instances, f"height/{scale}", f"width/{scale}"),
        }

    def get_saliency(self, inputs: List[Tensor]) -> Tensor:
        return reduce(self.forward(inputs)[3], "b i h w -> b h w", "max")

    def get_offsets_and_scales(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        device = inputs[0].device
        offsets, scales = [], []
        for x in inputs:
            h, w = x.shape[2:]
            y_min, x_min = 1 / h / 2, 1 / w / 2
            ys = torch.linspace(y_min, 1 - y_min, steps=h, device=device)
            xs = torch.linspace(x_min, 1 - x_min, steps=w, device=device)
            coordinate_grid = torch.stack(
                [repeat(xs, "w -> h w", h=h), repeat(ys, "h -> h w", w=w)], dim=2
            )
            offsets.append(coordinate_grid)
            cell_bbox = torch.tensor([-x_min, -y_min, x_min, y_min], device=device)
            scales.append(repeat(cell_bbox, "c -> h w c", h=h, w=w))
        return offsets, scales

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, _, full_height, full_width = inputs[0].shape
        feats = [
            lateral(inputs[level]) for level, lateral in zip(self.levels, self.laterals)
        ]
        flat_feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], 1)

        # compute locations
        loc_logits = self.loc_head(flat_feats).squeeze(2)
        loc_logits, loc_idxs = loc_logits.topk(self.max_instances, dim=1)
        batches = repeat(torch.arange(batch_size), f"b -> b {self.max_instances}")
        scores = loc_logits.sigmoid()
        num_instances = reduce(scores > 0.5, "b i -> b", "sum")
        flat_feats = flat_feats[batches, loc_idxs]

        mask_feats = self.mask_head(self.mask_lateral(inputs[self.mask_level]))
        mask_feats = repeat(mask_feats, "b c h w -> b i c h w", i=self.max_instances)

        offsets, _ = self.get_offsets_and_scales(
            inputs[self.bottom_level : self.top_level + 1]
        )
        mask_offsets = [repeat(_, "h w c -> b (h w) c", b=batch_size) for _ in offsets]
        mask_offsets = torch.cat(mask_offsets, dim=1)[batches, loc_idxs]
        mask_offsets = rearrange(mask_offsets, "b i c -> b i c 1 1")
        grid, _ = self.get_offsets_and_scales(
            inputs[self.mask_level : self.mask_level + 1]
        )
        grid = repeat(grid[0], "h w c -> b i c h w", b=batch_size, i=self.max_instances)
        grid = grid - mask_offsets
        mask_feats = torch.cat([mask_feats, grid], dim=2)
        dynamic_weights = self.kernel_head(flat_feats)

        c = self.mask_num_channels
        w1 = dynamic_weights[..., s := slice(0, (c + 2) * c)]
        w1 = w1.reshape(batch_size, self.max_instances, c + 2, c)
        b1 = dynamic_weights[..., s := slice(s.stop, s.stop + c)]
        b1 = b1.reshape(batch_size, self.max_instances, c, 1, 1)
        w2 = dynamic_weights[..., s := slice(s.stop, s.stop + c * c)]
        w2 = w2.reshape(batch_size, self.max_instances, c, c)
        b2 = dynamic_weights[..., s := slice(s.stop, s.stop + c)]
        b2 = b2.reshape(batch_size, self.max_instances, c, 1, 1)
        w3 = dynamic_weights[..., s := slice(s.stop, s.stop + c)]
        w3 = w3.reshape(batch_size, self.max_instances, c, 1)
        b3 = dynamic_weights[..., s.stop :]
        b3 = b3.reshape(batch_size, self.max_instances, 1, 1, 1)

        masks_preds = torch.einsum("bichw,bicd->bidhw", mask_feats, w1) + b1
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bichw,bicd->bidhw", masks_preds, w2) + b2
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bichw,bicd->bidhw", masks_preds, w3) + b3
        masks_preds = masks_preds.squeeze(2).sigmoid()

        class_logits = self.cls_head(flat_feats)
        classes = class_logits.max(dim=2).indices
        masks_preds = functional.interpolate(
            masks_preds, size=(full_height, full_width), mode="bilinear"
        )
        return num_instances, scores, classes, masks_preds

    def training_step(
        self,
        inputs: List[Tensor],
        classes: List[Tensor],
        masks: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert len(inputs) > self.top_level, "too few input levels"
        device = inputs[0].device
        batch_size, _, full_height, full_width = inputs[0].shape

        # remove degenerate instances (empty masks)
        valid_masks = [m.any((1, 2)) if m.shape[0] > 0 else None for m in masks]
        classes = [_classes[valid] for valid, _classes in zip(valid_masks, classes)]
        masks = [_masks[valid] for valid, _masks in zip(valid_masks, masks)]

        # compute anchors
        offsets, scales = self.get_offsets_and_scales(
            inputs[self.bottom_level : self.top_level + 1]
        )
        flat_offsets = torch.cat(
            [repeat(_, "h w c -> (h w) (2 c)", c=2) for _ in offsets]
        )
        flat_scales = torch.cat([rearrange(_, "h w c -> (h w) c") for _ in scales])
        full_size = torch.tensor([[full_width, full_height, full_width, full_height]])
        anchors = (flat_offsets + flat_scales) * full_size.to(device)

        # assign anchors to gt objects
        boxes = [ops.masks_to_boxes(mask) for mask in masks]
        matching_results = [
            ObjectDetection.bbox_matching(anchors, boxes[b], self.topk, relative=True)
            for b in range(batch_size)
        ]
        assignment = torch.stack([_[0] for _ in matching_results])
        rel_iou = torch.stack([_[1] for _ in matching_results])

        feats = [lateral(inputs[lv]) for lv, lateral in zip(self.levels, self.laterals)]
        flat_feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], 1)
        o2m_mask = rel_iou > 0
        o2m_weights = rel_iou[o2m_mask]
        o2m_feats = flat_feats[o2m_mask]

        # location loss
        loc_logits = self.loc_head(flat_feats).squeeze(2)
        with torch.autocast(device_type="cuda", enabled=False):
            loc_target = (rel_iou == 1.0).to(torch.float32)
            loc_loss = functional.binary_cross_entropy_with_logits(
                loc_logits.to(torch.float32), loc_target, reduction="none"
            )
            loc_loss = loc_loss.sum() / loc_target.sum()

        if rel_iou.max() == 0:
            return loc_loss, {
                "location_loss": loc_loss,
                "mask_loss": torch.zeros_like(loc_loss),
                "class_loss": torch.zeros_like(loc_loss),
            }

        # mask loss
        mask_feats = self.mask_head(self.mask_lateral(inputs[self.mask_level]))
        grid, _ = self.get_offsets_and_scales(
            inputs[self.mask_level : self.mask_level + 1]
        )
        biased_mask_feats = []
        for batch_idx in range(batch_size):
            loc_idxs = o2m_mask[batch_idx].nonzero()[:, 0]
            n_objects = loc_idxs.shape[0]
            if n_objects == 0:
                continue
            _mask_feats = repeat(mask_feats[batch_idx], "c h w -> i c h w", i=n_objects)
            _rel_offsets = rearrange(flat_offsets[loc_idxs], "i c -> i c 1 1")
            _grid = (
                repeat(grid[0], "h w c ->  i c h w", i=n_objects) - _rel_offsets[:, :2]
            )
            _mask_feats = torch.cat([_mask_feats, _grid], dim=1)
            biased_mask_feats.append(_mask_feats)

        if len(biased_mask_feats) == 0:
            return loc_loss, {
                "location_loss": loc_loss,
                "mask_loss": torch.zeros_like(loc_loss),
                "class_loss": torch.zeros_like(loc_loss),
            }

        biased_mask_feats = torch.cat(biased_mask_feats)
        dyn_weights = self.kernel_head(o2m_feats)
        n_obj = dyn_weights.shape[0]
        c = self.mask_num_channels
        w1 = dyn_weights[:, (s := slice(0, (c + 2) * c))].reshape(n_obj, c + 2, c)
        b1 = dyn_weights[:, (s := slice(s.stop, s.stop + c))].reshape(n_obj, c, 1, 1)
        w2 = dyn_weights[:, (s := slice(s.stop, s.stop + c * c))].reshape(n_obj, c, c)
        b2 = dyn_weights[:, (s := slice(s.stop, s.stop + c))].reshape(n_obj, c, 1, 1)
        w3 = dyn_weights[:, (s := slice(s.stop, s.stop + c))]
        w3 = w3.reshape(n_obj, c, 1)
        b3 = dyn_weights[:, s.stop :].reshape(n_obj, 1, 1, 1)

        masks_preds = torch.einsum("bchw,bcd->bdhw", biased_mask_feats, w1) + b1
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bchw,bcd->bdhw", masks_preds, w2) + b2
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bchw,bcd->bdhw", masks_preds, w3) + b3
        masks_preds = masks_preds.squeeze(1).sigmoid()

        target_masks = torch.cat(
            [masks[b][assignment[b, o2m_mask[b]]] for b in range(batch_size)]
        ).to(masks_preds)
        target_masks = functional.interpolate(
            target_masks.unsqueeze(1), size=masks_preds.shape[1:], mode="bilinear"
        ).squeeze(1)
        # dice loss
        numerator = reduce(masks_preds * target_masks, "c h w -> c", "sum")
        denominator = reduce(masks_preds**2 + target_masks**2, "c h w -> c", "sum")
        with torch.autocast(device_type="cuda", enabled=False):
            mask_loss = 1 - 2 * numerator.to(torch.float32) / denominator
            mask_loss = (o2m_weights * mask_loss).sum() / o2m_weights.sum()

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

        loss = loc_loss + 10 * mask_loss + class_loss
        metrics = {
            "location_loss": loc_loss,
            "mask_loss": mask_loss,
            "class_loss": class_loss,
        }
        return loss, metrics

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        max_detection_thresholds = [1, min(self.max_instances, 10), self.max_instances]
        self.map_computer = MeanAveragePrecision(
            max_detection_thresholds=max_detection_thresholds,
            iou_type="segm",
            backend="faster_coco_eval",
        )

    def validation_step(
        self, inputs: List[Tensor], classes: List[Tensor], masks: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        _, scores, pred_classes, pred_masks = self.forward(inputs)
        self.map_computer.to(scores.device).update(
            [
                {"scores": s, "labels": c, "masks": m > 0.5}
                for s, c, m in zip(scores, pred_classes, pred_masks)
            ],
            [{"labels": c, "masks": m > 0.5} for c, m in zip(classes, masks)],
        )
        loss, metrics = self.training_step(inputs, classes, masks, is_validating=True)
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

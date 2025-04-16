from functools import partial
from typing import Any, Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchvision import ops
import torch

from sihl.utils import coordinate_grid, ObjectKeypointSimilarity
from sihl.heads.object_detection import ObjectDetection


class KeypointDetection(ObjectDetection):
    """Keypoint detection is the prediction of ..."""

    def __init__(
        self,
        in_channels: List[int],
        num_keypoints: int,
        bottom_level: int = 3,
        top_level: int = 5,
        num_channels: int = 256,
        num_layers: int = 4,
        max_instances: int = 100,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_keypoints (int): Number of possible keypoints per instance.
            bottom_level (int, optional): Bottom level of inputs this head is attached to. Defaults to 3.
            top_level (int, optional): Top level of inputs this head is attached to. Defaults to 5.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 4.
            max_instances (int, optional): Maximum number of instances to predict in a sample. Defaults to 100.
        """
        super().__init__(
            in_channels=in_channels,
            num_classes=1,
            bottom_level=bottom_level,
            top_level=top_level,
            num_channels=num_channels,
            num_layers=num_layers,
            max_instances=max_instances,
        )
        self.num_keypoints = num_keypoints
        MLP = partial(ops.MLP, norm_layer=nn.LayerNorm, activation_layer=nn.SiLU)
        Conv = partial(ops.Conv2dNormActivation, activation_layer=nn.SiLU)

        c = self.mask_num_channels = 32  # 8
        kernel_params = (c + 2) * c + c + c * c + c + c * num_keypoints + num_keypoints
        self.kernel_head = MLP(
            num_channels, [num_channels for _ in range(num_layers)] + [kernel_params]
        )
        self.mask_laterals = nn.ModuleList(
            [Conv(in_channels[lvl], num_channels, 1) for lvl in self.levels]
        )
        self.mask_stem = Conv(num_channels, self.mask_num_channels, 3)
        self.output_shapes = {
            "num_instances": ("batch_size",),
            "instance_scores": ("batch_size", max_instances),
            "keypoint_scores": ("batch_size", max_instances, num_keypoints),
            "keypoints": ("batch_size", max_instances, num_keypoints, 2),
        }

    def forward(
        self, inputs: List[Tensor], faster: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        device = inputs[0].device
        batch_size, _, full_height, full_width = inputs[0].shape
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
            w3 = w[..., (s := slice(s.stop, s.stop + c * self.num_keypoints))]
            w3 = w3.reshape(n_groups * self.num_keypoints, c, 1, 1)
            b3 = w[..., s.stop :].reshape(n_groups * self.num_keypoints)
            masks = functional.silu(torch.conv2d(mask_feats, w1, b1, groups=n_groups))
            masks = functional.silu(torch.conv2d(masks, w2, b2, groups=n_groups))
            masks = functional.sigmoid(torch.conv2d(masks, w3, b3, groups=n_groups))[0]
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
                w3 = w[..., (s := slice(s.stop, s.stop + c * self.num_keypoints))]
                w3 = w3.reshape(n_groups * self.num_keypoints, c, 1, 1)
                b3 = w[..., s.stop :].reshape(n_groups * self.num_keypoints)
                masks = functional.silu(
                    torch.conv2d(mask_feats[b : b + 1], w1, b1, groups=n_groups)
                )
                masks = functional.silu(torch.conv2d(masks, w2, b2, groups=n_groups))
                masks = torch.conv2d(masks, w3, b3, groups=n_groups)
                masks_list.append(masks[0])
            masks = torch.cat(masks_list).sigmoid()
        masks = rearrange(
            masks,
            "(b i k) h w -> b i k (h w)",
            b=batch_size,
            i=self.max_instances,
            k=self.num_keypoints,
        )
        keypoint_scores, flat_idxs = masks.max(dim=3)
        kpts_y, kpts_x = flat_idxs // mask_height, flat_idxs % mask_height
        kpts_y = kpts_y / mask_height * full_height
        kpts_x = kpts_x / mask_width * full_width
        keypoints = torch.stack([kpts_x, kpts_y], dim=3)
        return num_instances, scores, keypoint_scores, keypoints

    def training_step(
        self,
        inputs: List[Tensor],
        keypoints: List[Tensor],
        presence: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        assert len(inputs) > self.top_level, "too few input levels"
        device = inputs[0].device
        batch_size, _, full_height, full_width = inputs[0].shape

        # remove degenerate instances (those with no visible keypoint)
        for b in range(batch_size):
            non_degenerate_instances = presence[b].any(dim=1)
            keypoints[b] = keypoints[b][non_degenerate_instances]
            presence[b] = presence[b][non_degenerate_instances]

        boxes = [self.keypoints_to_boxes(*args) for args in zip(keypoints, presence)]
        classes = [torch.zeros_like(b, dtype=torch.int64)[:, 0] for b in boxes]
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

        if not torch.is_nonzero(o2m_mask.sum()):
            metrics["keypoint_loss"] = metrics["position_loss"] = 0
            return loss, metrics

        feats = self.get_features(inputs)
        flat_feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], 1)

        mask_height, mask_width = inputs[self.bottom_level].shape[2:]
        mask_feats = [
            functional.interpolate(x, (mask_height, mask_width), mode="bilinear")
            for x in feats
        ]
        mask_feats = self.mask_stem(torch.stack(mask_feats).sum(dim=0))
        rel_offsets = [
            coordinate_grid(*inputs[lvl].shape[2:]).to(device) for lvl in self.levels
        ]
        rel_offsets = torch.cat([rearrange(_, "h w c -> (h w) c") for _ in rel_offsets])
        grid = coordinate_grid(height=mask_height, width=mask_width).to(device)
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
        w3 = dyn_weights[:, (s := slice(s.stop, s.stop + c * self.num_keypoints))]
        w3 = w3.reshape(n_obj * self.num_keypoints, c, 1, 1)
        b3 = dyn_weights[:, s.stop :].reshape(n_obj * self.num_keypoints)

        masks_preds = functional.silu(
            torch.conv2d(biased_mask_feats, w1, b1, groups=n_obj)
        )
        masks_preds = functional.silu(torch.conv2d(masks_preds, w2, b2, groups=n_obj))
        masks_preds = torch.conv2d(masks_preds, w3, b3, groups=n_obj)[0]

        masks_preds = rearrange(
            masks_preds, "(i k) h w -> i k h w", i=n_obj, k=self.num_keypoints
        )
        target_keypoints = torch.cat(
            [keypoints[b][assignment[b, o2m_mask[b]]] for b in range(batch_size)]
        )
        target_presence = torch.cat(
            [presence[b][assignment[b, o2m_mask[b]]] for b in range(batch_size)]
        )
        target_heatmaps = self.keypoints_to_heatmaps(
            target_keypoints,
            target_presence,
            height=mask_height,
            width=mask_width,
            img_height=full_height,
            img_width=full_width,
        )
        with torch.autocast(device_type="cuda", enabled=False):
            keypoint_loss = ops.sigmoid_focal_loss(
                masks_preds, target_heatmaps, reduction="none"
            )
            o2m_weight = o2m_weight[:, None, None, None]
            keypoint_loss = (o2m_weight * keypoint_loss).sum() / o2m_weight.sum()

            keypoint_scores, flat_idxs = masks_preds.flatten(2, 3).max(dim=2)
            kpts_y, kpts_x = flat_idxs // mask_height, flat_idxs % mask_height
            kpts_y = kpts_y / mask_height
            kpts_x = kpts_x / mask_width
            pred_keypoints = torch.stack([kpts_x, kpts_y], dim=2)
            size = torch.tensor([[[full_width, full_height]]], device=device)
            position_loss = target_presence.unsqueeze(2) * functional.mse_loss(
                pred_keypoints, target_keypoints / size, reduction="none"
            )
            position_loss = (o2m_weight * keypoint_loss).sum() / o2m_weight.sum()

        loss = loss + keypoint_loss + position_loss
        metrics["keypoint_loss"] = keypoint_loss
        metrics["position_loss"] = position_loss
        return loss, metrics

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.oks_computer = ObjectKeypointSimilarity()

    def validation_step(
        self, inputs: List[Tensor], keypoints: List[Tensor], presence: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        batch_size = inputs[0].shape[0]
        num_instances, scores, keypoint_scores, pred_keypoints = self.forward(inputs)
        self.oks_computer.update(
            preds=[
                {
                    "scores": scores[b],
                    "keypoints": torch.cat(
                        [
                            pred_keypoints[b][:, :, :],
                            keypoint_scores[b][:, :, None] > 0.5,
                        ],
                        dim=2,
                    ),
                }
                for b in range(batch_size)
            ],
            targets=[
                {
                    "keypoints": torch.cat(
                        [keypoints[b][:, :, :], presence[b][:, :, None]], dim=2
                    )
                }
                for b in range(batch_size)
            ],
        )
        loss, metrics = self.training_step(
            inputs, keypoints, presence, is_validating=True
        )
        self.loss_computer.to(loss.device).update(loss)
        return loss, metrics

    def on_validation_end(self) -> Dict[str, Any]:
        metrics = self.oks_computer.compute()
        metrics["loss"] = self.loss_computer.compute()
        return metrics

    @staticmethod
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

    @staticmethod
    def keypoints_to_heatmaps(
        keypoints: Tensor,
        presence: Tensor,
        height: int,
        width: int,
        img_height: int,
        img_width: int,
    ) -> Tensor:
        heatmaps = keypoints.clone()
        heatmaps[:, :, 0] *= (width - 1) / (img_width - 1)
        heatmaps[:, :, 0].clamp_(0, width - 1)
        heatmaps[:, :, 1] *= (height - 1) / (img_height - 1)
        heatmaps[:, :, 1].clamp_(0, height - 1)
        heatmaps = heatmaps.round().to(torch.int64)
        gt_xs = functional.one_hot(heatmaps[:, :, 0], width).to(torch.float32)
        gt_ys = functional.one_hot(heatmaps[:, :, 1], height).to(torch.float32)
        heatmaps = presence[:, :, None, None] * gt_xs.unsqueeze(2) * gt_ys.unsqueeze(3)
        return heatmaps

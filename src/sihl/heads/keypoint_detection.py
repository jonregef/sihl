from functools import partial
from typing import Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchvision import ops
import torch

from sihl.utils import ObjectKeypointSimilarity


class KeypointDetection(nn.Module):
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
            num_keypoints (int): Number of keypoints.
            bottom_level (int, optional): Bottom level of inputs this head is attached to. Defaults to 3.
            top_level (int, optional): Top level of inputs this head is attached to. Defaults to 5.
            num_channels (int, optional): Number of convolutional channels. Defaults to 256.
            num_layers (int, optional): Number of convolutional layers. Defaults to 4.
            max_instances (int, optional): Maximum number of instances to predict in a sample. Defaults to 100.
        """
        assert num_keypoints > 0, num_keypoints
        assert len(in_channels) > top_level, (len(in_channels), top_level)
        assert 0 < bottom_level <= top_level, (bottom_level, top_level)
        assert num_channels % 4 == 0, num_channels
        assert num_layers >= 0, num_layers
        assert max_instances > 0, max_instances
        super().__init__()

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
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

        c = self.mask_num_channels = 8
        kernel_params = (c + 2) * c + c + c * c + c + c * num_keypoints + num_keypoints
        self.kernel_head = MLP(num_channels, hidden_channels + [kernel_params])
        self.mask_laterals = nn.ModuleList(
            [Conv(in_channels[level], num_channels, 1) for level in self.levels]
        )
        self.mask_head = Conv(num_channels, self.mask_num_channels, 3)

        self.output_shapes = {
            "num_instances": ("batch_size",),
            "scores": ("batch_size", max_instances),
            "presence": ("batch_size", max_instances, num_keypoints),
            "keypoints": ("batch_size", max_instances, num_keypoints, 2),
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
        for level in range(self.bottom_level, self.top_level + 1):
            h, w = inputs[level].shape[2:]
            y_min, x_min = 1 / h / 2, 1 / w / 2
            ys = torch.linspace(y_min, 1 - y_min, steps=h, device=device)
            xs = torch.linspace(x_min, 1 - x_min, steps=w, device=device)
            coordinate_grid = torch.stack(
                [repeat(xs, "w -> h w", h=h), repeat(ys, "h -> h w", w=w)], dim=2
            )
            # rel_grid = rearrange(coordinate_grid, "h w c -> (h w) c")
            rel_offsets.append(coordinate_grid)
            levels.append(torch.full((h * w, 1), level, device=device))
        # rel_offsets, levels = torch.cat(rel_offsets), torch.cat(levels)
        # rel_offsets = repeat(rel_offsets, "i c -> i (2 c)", c=2)
        return rel_offsets, levels

    def get_features(self, inputs: List[Tensor]) -> List[Tensor]:
        global_context = self.global_context(inputs[self.top_level])
        return [
            lateral(inputs[self.bottom_level + idx]) + global_context
            for idx, lateral in enumerate(self.laterals)
        ]

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, _, full_height, full_width = inputs[0].shape
        feats = self.get_features(inputs)
        flat_feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], 1)
        # compute locations
        loc_logits = self.loc_head(flat_feats).squeeze(2)
        loc_logits, loc_idxs = loc_logits.topk(self.max_instances, dim=1)
        batches = repeat(torch.arange(batch_size), f"b -> b {self.max_instances}")
        scores = loc_logits.sigmoid()
        num_instances = reduce(scores > 0.5, "b i -> b", "sum")

        flat_feats = flat_feats[batches, loc_idxs]

        mask_height, mask_width = inputs[self.bottom_level].shape[2:]
        mask_feats = [
            functional.interpolate(x, size=(mask_height, mask_width), mode="bilinear")
            for x in feats
        ]
        mask_feats = self.mask_head(torch.stack(mask_feats).sum(0))
        mask_feats = repeat(mask_feats, "b c h w -> b i c h w", i=self.max_instances)

        rel_offsets, levels = self.get_offsets_and_levels(inputs)
        mask_offsets = torch.cat(
            [repeat(_, "h w c -> b (h w) c", b=batch_size) for _ in rel_offsets], dim=1
        )[batches, loc_idxs]
        mask_offsets = rearrange(mask_offsets, "b i c -> b i c 1 1")
        grid = repeat(
            rel_offsets[0], "h w c -> b i c h w", b=batch_size, i=self.max_instances
        )
        grid = grid - mask_offsets
        mask_feats = torch.cat([mask_feats, grid], dim=2)
        dynamic_weights = self.kernel_head(flat_feats)

        c = self.mask_num_channels
        w1 = dynamic_weights[..., s := slice(0, (c + 2) * c)].reshape(
            batch_size, self.max_instances, c + 2, c
        )
        b1 = dynamic_weights[..., s := slice(s.stop, s.stop + c)].reshape(
            batch_size, self.max_instances, c, 1, 1
        )
        w2 = dynamic_weights[..., s := slice(s.stop, s.stop + c * c)].reshape(
            batch_size, self.max_instances, c, c
        )
        b2 = dynamic_weights[..., s := slice(s.stop, s.stop + c)].reshape(
            batch_size, self.max_instances, c, 1, 1
        )
        w3 = dynamic_weights[
            ..., s := slice(s.stop, s.stop + c * self.num_keypoints)
        ].reshape(batch_size, self.max_instances, c, self.num_keypoints)
        b3 = dynamic_weights[..., s.stop :].reshape(
            batch_size, self.max_instances, self.num_keypoints, 1, 1
        )

        masks_preds = torch.einsum("bichw,bicd->bidhw", mask_feats, w1) + b1
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bichw,bicd->bidhw", masks_preds, w2) + b2
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bichw,bicd->bidhw", masks_preds, w3) + b3
        masks_preds = functional.sigmoid(masks_preds)  # .squeeze(2)

        presence, flat_idxs = masks_preds.flatten(3, 4).max(3)
        kpts_y, kpts_x = flat_idxs // mask_height, flat_idxs % mask_height
        kpts_y = kpts_y.to(torch.float32) / mask_height * full_height
        kpts_x = kpts_x.to(torch.float32) / mask_width * full_width
        keypoints = torch.stack([kpts_x, kpts_y], dim=3)

        return num_instances, scores, presence, keypoints

    def training_step(
        self,
        inputs: List[Tensor],
        presence: List[Tensor],
        keypoints: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        assert len(inputs) > self.top_level, "too few input levels"
        device = inputs[0].device
        batch_size, _, full_height, full_width = inputs[0].shape
        full_size = torch.tensor(
            [[full_width, full_height, full_width, full_height]], device=device
        )

        # remove degenerate instances (those with no visible keypoint) FIXME: mutation
        for b in range(batch_size):
            non_degenerate_instances = presence[b].any(dim=1)
            keypoints[b] = keypoints[b][non_degenerate_instances]
            presence[b] = presence[b][non_degenerate_instances]

        boxes = [self.keypoints_to_boxes(*args) for args in zip(keypoints, presence)]

        # compute anchors
        rel_offsets, levels = self.get_offsets_and_levels(inputs)
        flat_offsets = [rearrange(_, "h w c -> (h w) c") for _ in rel_offsets]
        flat_offsets, levels = torch.cat(flat_offsets), torch.cat(levels)
        flat_offsets = repeat(flat_offsets, "i c -> i (2 c)", c=2)
        directions = torch.tensor([[-1, -1, 1, 1]], device=device)
        scale = torch.sigmoid(levels - self.top_level)
        anchors = (flat_offsets + directions * scale) * full_size

        # assign anchors to gt objects
        matching_results = [
            self.bbox_matching(anchors, boxes[b], self.topk) for b in range(batch_size)
        ]
        assignment = torch.stack([_[0] for _ in matching_results])
        o2o_mask = torch.stack([_[1] for _ in matching_results])
        # o2m_iou = torch.stack([_[2] for _ in matching_results])
        rel_iou = torch.stack([_[3] for _ in matching_results])

        feats = self.get_features(inputs)
        flat_feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], 1)
        o2m_mask = rel_iou > 0
        loc_target = rel_iou / self.topk
        loc_target[o2o_mask] = 1
        o2m_weights = rel_iou[o2m_mask]
        o2m_feats = flat_feats[o2m_mask]

        # compute location loss
        loc_logits = self.loc_head(flat_feats).squeeze(2)
        with torch.autocast(device_type="cuda", enabled=False):
            loc_loss = functional.binary_cross_entropy_with_logits(
                loc_logits.to(torch.float32), loc_target, reduction="none"
            )
            loc_loss = loc_loss.sum() / loc_target.sum()

        mask_height, mask_width = inputs[self.bottom_level].shape[2:]
        mask_feats = [
            functional.interpolate(x, size=(mask_height, mask_width), mode="bilinear")
            for x in feats
        ]
        mask_feats = self.mask_head(torch.stack(mask_feats).sum(dim=0))
        grid = rel_offsets[0]
        biased_mask_feats = []
        for batch_idx in range(batch_size):
            loc_idxs = o2m_mask[batch_idx].nonzero()[:, 0]
            n_objects = loc_idxs.shape[0]
            if n_objects == 0:
                continue
            _mask_feats = repeat(mask_feats[batch_idx], "c h w -> i c h w", i=n_objects)
            _rel_offsets = rearrange(flat_offsets[loc_idxs], "i c -> i c 1 1")
            _grid = repeat(grid, "h w c ->  i c h w", i=n_objects) - _rel_offsets[:, :2]
            _mask_feats = torch.cat([_mask_feats, _grid], dim=1)
            # _mask_feats = rearrange(_mask_feats, "i c h w -> (i c) h w")
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
        w3 = dyn_weights[
            :, (s := slice(s.stop, s.stop + c * self.num_keypoints))
        ].reshape(n_obj, c, self.num_keypoints)
        b3 = dyn_weights[:, s.stop :].reshape(n_obj, self.num_keypoints, 1, 1)

        masks_preds = torch.einsum("bchw,bcd->bdhw", biased_mask_feats, w1) + b1
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bchw,bcd->bdhw", masks_preds, w2) + b2
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bchw,bcd->bdhw", masks_preds, w3) + b3

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
            keypoint_loss = (
                o2m_weights.reshape(-1, 1, 1, 1) * keypoint_loss
            ).sum() / o2m_weights.sum()

            keypoint_scores, flat_idxs = masks_preds.flatten(2, 3).max(dim=2)
            kpts_y, kpts_x = flat_idxs // mask_height, flat_idxs % mask_height
            kpts_y = kpts_y / mask_height
            kpts_x = kpts_x / mask_width
            pred_keypoints = torch.stack([kpts_x, kpts_y], dim=2)
            size = torch.tensor([[[full_width, full_height]]], device=device)
            position_loss = target_presence.unsqueeze(2) * functional.mse_loss(
                pred_keypoints, target_keypoints / size, reduction="none"
            )
            position_loss = (o2m_weights * keypoint_loss).sum() / o2m_weights.sum()

        loss = loc_loss + keypoint_loss + position_loss
        metrics = {
            "location_loss": loc_loss,
            "keypoint_loss": keypoint_loss,
            "position_loss": position_loss,
        }
        return loss, metrics

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
                if keypoints[b].shape[0] > 0
            ],
            targets=[
                {
                    "keypoints": torch.cat(
                        [keypoints[b][:, :, :], presence[b][:, :, None]], dim=2
                    )
                }
                for b in range(batch_size)
                if keypoints[b].shape[0] > 0
            ],
        )
        loss, metrics = self.training_step(
            inputs, keypoints=keypoints, presence=presence, is_validating=True
        )
        self.loss_computer.to(loss.device).update(loss)
        return loss, metrics

    def on_validation_end(self) -> Dict[str, float]:
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

from functools import partial
from typing import Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import MeanMetric
from torchvision import ops
import torch

from sihl.heads.object_detection import ObjectDetection
from sihl.utils import PercentageOfCorrectKeypoints


class KeypointDetection(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_keypoints: int,
        mask_level: int = 3,
        bottom_level: int = 5,
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
        self.presence_head = MLP(num_channels, hidden_channels + [num_keypoints])

        c = self.mask_num_channels = 32
        kernel_params = (c + 2) * c + c + c * c + c + c * num_keypoints + num_keypoints
        self.kernel_head = MLP(num_channels, hidden_channels + [kernel_params])
        self.mask_lateral = Conv(in_channels[self.mask_level], num_channels, 1)
        self.mask_head = ops.Conv2dNormActivation(
            num_channels, c, 3, activation_layer=nn.SiLU
        )

        self.output_shapes = {
            "num_instances": ("batch_size",),
            "scores": ("batch_size", max_instances),
            "presence": ("batch_size", max_instances, num_keypoints),
            "keypoints": ("batch_size", max_instances, num_keypoints, 2),
        }

    def get_saliency(self, inputs: List[Tensor]) -> Tensor:
        return reduce(
            self.forward(inputs, output_heatmaps=True), "b i c h w -> b h w", "max"
        )

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

    def forward(
        self, inputs: List[Tensor], output_heatmaps: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
        mask_offsets = torch.cat(
            [repeat(_, "h w c -> b (h w) c", b=batch_size) for _ in offsets], dim=1
        )[batches, loc_idxs]
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
        w3 = dynamic_weights[..., s := slice(s.stop, s.stop + c * self.num_keypoints)]
        w3 = w3.reshape(batch_size, self.max_instances, c, self.num_keypoints)
        b3 = dynamic_weights[..., s.stop :]
        b3 = b3.reshape(batch_size, self.max_instances, self.num_keypoints, 1, 1)

        masks_preds = torch.einsum("bichw,bicd->bidhw", mask_feats, w1) + b1
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bichw,bicd->bidhw", masks_preds, w2) + b2
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bichw,bicd->bidhw", masks_preds, w3) + b3
        presence = self.presence_head(flat_feats).sigmoid()

        if output_heatmaps:
            return masks_preds.flatten(3, 4).softmax(3).reshape(masks_preds.shape)
        else:
            mask_height, mask_width = masks_preds.shape[3:]
            _, flat_idxs = masks_preds.flatten(3, 4).max(3)
            kpts_y, kpts_x = flat_idxs // mask_height, flat_idxs % mask_height
            kpts_y = (kpts_y.to(torch.float32) + 0.5) / mask_height * full_height
            kpts_x = (kpts_x.to(torch.float32) + 0.5) / mask_width * full_width
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

        # remove degenerate instances (those with no visible keypoint)
        clean_keypoints, clean_presence = [], []
        for b in range(batch_size):
            non_degenerate_instances = presence[b].any(dim=1)
            clean_keypoints.append(keypoints[b][non_degenerate_instances])
            clean_presence.append(presence[b][non_degenerate_instances])
        boxes = [
            self.keypoints_to_boxes(kpts, pres)
            for kpts, pres in zip(clean_keypoints, clean_presence)
        ]

        # compute anchors
        offsets, scales = self.get_offsets_and_scales(
            inputs[self.bottom_level : self.top_level + 1]
        )
        flat_offsets = torch.cat(
            [repeat(_, "h w c -> (h w) (2 c)", c=2) for _ in offsets]
        )
        flat_scales = torch.cat([rearrange(_, "h w c -> (h w) c") for _ in scales])
        anchors = (flat_offsets + flat_scales) * full_size

        # assign anchors to gt objects
        matching_results = [
            ObjectDetection.bbox_matching(anchors, boxes[b], self.topk, relative=True)
            for b in range(batch_size)
        ]
        assignment = torch.stack([_[0] for _ in matching_results])
        rel_iou = torch.stack([_[1] for _ in matching_results])

        feats = [
            lateral(inputs[level]) for level, lateral in zip(self.levels, self.laterals)
        ]
        flat_feats = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in feats], 1)
        o2m_mask = rel_iou > 0
        o2m_weights = rel_iou[o2m_mask].reshape(-1, 1)
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
            zero = torch.zeros_like(loc_loss)
            metrics = {
                "location_loss": loc_loss,
                **{"box_loss": zero, "class_loss": zero, "iou_loss": zero},
            }
            return loc_loss, metrics

        # presence loss
        presence_logits = self.presence_head(o2m_feats)
        target_presence = torch.cat(
            [clean_presence[b][assignment[b, o2m_mask[b]]] for b in range(batch_size)]
        )
        with torch.autocast(device_type="cuda", enabled=False):
            presence_loss = functional.binary_cross_entropy_with_logits(
                presence_logits, target_presence.to(torch.float32), reduction="none"
            )
            presence_loss = (o2m_weights * presence_loss).sum() / o2m_weights.sum()

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
        w3 = dyn_weights[:, (s := slice(s.stop, s.stop + c * self.num_keypoints))]
        w3 = w3.reshape(n_obj, c, self.num_keypoints)
        b3 = dyn_weights[:, s.stop :].reshape(n_obj, self.num_keypoints, 1, 1)

        masks_preds = torch.einsum("bchw,bcd->bdhw", biased_mask_feats, w1) + b1
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bchw,bcd->bdhw", masks_preds, w2) + b2
        masks_preds = functional.silu(masks_preds)
        masks_preds = torch.einsum("bchw,bcd->bdhw", masks_preds, w3) + b3

        target_keypoints = torch.cat(
            [clean_keypoints[b][assignment[b, o2m_mask[b]]] for b in range(batch_size)]
        )
        target_heatmaps = self.keypoints_to_heatmaps(
            target_keypoints,
            target_presence,
            height=mask_feats.shape[2],
            width=mask_feats.shape[3],
            img_height=full_height,
            img_width=full_width,
        )
        with torch.autocast(device_type="cuda", enabled=False):
            masks_preds = rearrange(masks_preds, "b c h w -> b (h w) c")
            target_heatmaps = rearrange(target_heatmaps, "b c h w -> b (h w) c")
            keypoint_loss = functional.cross_entropy(
                masks_preds, target_heatmaps, reduction="none"
            )
            keypoint_loss = (o2m_weights * keypoint_loss).sum() / o2m_weights.sum()

        loss = loc_loss + keypoint_loss + presence_loss
        metrics = {
            "location_loss": loc_loss,
            "keypoint_loss": keypoint_loss,
            "presence_loss": presence_loss,
        }
        return loss, metrics

    def on_validation_start(self) -> None:
        self.loss_computer = MeanMetric(nan_strategy="ignore")
        self.pck_computer = PercentageOfCorrectKeypoints()

    def validation_step(
        self, inputs: List[Tensor], keypoints: List[Tensor], presence: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, float]]:
        batch_size, _, full_height, full_width = inputs[0].shape
        full_size = torch.tensor([[[full_width, full_height]]], device=inputs[0].device)
        num_instances, scores, keypoint_scores, pred_keypoints = self.forward(inputs)
        for b in range(batch_size):
            self.pck_computer.to(pred_keypoints.device).update(
                pred_keypoints=pred_keypoints[b, : num_instances[b]] / full_size,
                pred_presence=keypoint_scores[b, : num_instances[b]],
                gt_keypoints=keypoints[b] / full_size,
                gt_presence=presence[b],
            )
        loss, metrics = self.training_step(
            inputs, keypoints=keypoints, presence=presence, is_validating=True
        )
        self.loss_computer.to(loss.device).update(loss)
        return loss, metrics

    def on_validation_end(self) -> Dict[str, float]:
        metrics = self.pck_computer.compute()
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

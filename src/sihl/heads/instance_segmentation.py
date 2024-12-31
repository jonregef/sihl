from typing import Any, Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchvision.ops import masks_to_boxes
import torch

from sihl.layers import ConvNormAct
from sihl.utils import coordinate_grid, interpolate

from .object_detection import ObjectDetection


class InstanceSegmentation(ObjectDetection):
    """Instance segmentation is like object detection except instances are associated
    with a binary mask instead of a bounding box.

    Refs:
        1. [CondInst](https://arxiv.org/abs/2102.03026)
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
        mask_top_level: int = 5,
        mask_num_channels: int = 8,
        mask_out_channels: int = 1,
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
            mask_top_level (int, optional): Top level of inputs masks are computed from. Defaults to 5.
            mask_num_channels (int, optional): Number of convolutional channels in the mask prediction branch.
            mask_out_channels (int, optional): Number of output channels in the mask prediction branch.
        """
        assert top_level >= mask_top_level >= bottom_level
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            bottom_level=bottom_level,
            top_level=top_level,
            num_channels=num_channels,
            num_layers=num_layers,
            max_instances=max_instances,
            t_min=t_min,
            t_max=t_max,
            topk=topk,
            soft_label_decay_steps=soft_label_decay_steps,
        )

        self.mask_top_level = mask_top_level
        self.mask_num_channels = mask_num_channels
        self.mask_out_channels = mask_out_channels
        c, o = mask_num_channels, mask_out_channels
        controller_out_channels = (c + 2) * c + c + c * c + c + c * o + o
        self.controller_head = nn.Conv2d(num_channels, controller_out_channels, 1)
        bl = bottom_level
        self.bottom_module = ConvNormAct(in_channels[bl], self.mask_num_channels)
        self.output_shapes = {
            "num_instances": ("batch_size",),
            "scores": ("batch_size", max_instances),
            "classes": ("batch_size", max_instances),
            "masks": ("batch_size", max_instances, f"height/{2**bl}", f"width/{2**bl}"),
        }

    def forward(self, inputs: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_instances, scores, classes, logits = self.forward_logits(inputs)
        masks = interpolate(logits.sigmoid(), size=inputs[0].shape[2:])
        return num_instances, scores, classes, masks

    def forward_logits(
        self, inputs: List[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scores, reg_features = self.get_features(inputs)
        scores, pred_classes = scores.max(dim=2)
        # keep only the top `max_instances`
        batch_size = scores.shape[0]
        batches = repeat(torch.arange(batch_size), f"b -> b {self.max_instances}")
        instance_idxs = scores.topk(self.max_instances).indices
        scores = scores[batches, instance_idxs]
        pred_classes = pred_classes[batches, instance_idxs]
        num_instances = reduce(scores > self.threshold, "b i -> b", "sum")

        # compute segmentation masks for the `max_instances` top scoring anchors
        controllers = self.get_controllers(reg_features)
        bottom_features = self.get_bottom_features(inputs)
        normalized_anchors = self.get_anchors(inputs, normalized=True)
        logits = torch.cat(
            [
                self.get_masks(
                    params=controllers[batch_idx, instance_idxs[batch_idx]],
                    anchors=normalized_anchors[instance_idxs[batch_idx]],
                    bottom_features=bottom_features[batch_idx],
                    num_masks=self.max_instances,
                )
                for batch_idx in range(batch_size)
            ]
        )
        return num_instances, scores, pred_classes, logits

    def get_controllers(self, features: List[Tensor]) -> Tensor:
        controllers = [self.controller_head(_) for _ in features]
        return torch.cat(
            [rearrange(_, "n c h w -> n (h w) c") for _ in controllers], dim=1
        )

    def get_bottom_features(self, inputs: List[Tensor]) -> Tensor:
        assert len(inputs) > self.top_level
        x0 = inputs[self.bottom_level]
        batch_size, _, height, width = x0.shape
        summed_features = torch.stack(
            [x0]
            + [
                interpolate(inputs[level], size=(height, width))
                for level in range(self.bottom_level + 1, self.mask_top_level + 1)
            ]
        ).sum(dim=0)
        bottom_feats = self.bottom_module(summed_features)
        # Absolute coordinates will later be transformed into relative ones
        abs_coords = coordinate_grid(height, width, y_max=1, x_max=1)
        abs_coords = repeat(abs_coords, "c h w -> b c h w", b=batch_size, c=2)
        return torch.cat([bottom_feats, abs_coords.to(bottom_feats)], dim=1)

    def get_masks(
        self, params: Tensor, anchors: Tensor, bottom_features: Tensor, num_masks: int
    ) -> Tensor:
        c, m_c = self.mask_num_channels, num_masks * self.mask_num_channels
        o, m_o = self.mask_out_channels, num_masks * self.mask_out_channels

        w1 = params[:, (s := slice(0, (c + 2) * c))].reshape(m_c, c + 2, 1, 1)
        b1 = params[:, (s := slice(s.stop, s.stop + c))].reshape(m_c)
        w2 = params[:, (s := slice(s.stop, s.stop + c**2))].reshape(m_c, c, 1, 1)
        b2 = params[:, (s := slice(s.stop, s.stop + c))].reshape(m_c)
        w3 = params[:, (s := slice(s.stop, s.stop + c * o))].reshape(m_o, c, 1, 1)
        b3 = params[:, s.stop :].reshape(m_o)

        # Convert abs coords to rel coords by subtracting padded anchors
        padded_anchors = functional.pad(anchors, (c, 0, 0, 0)).reshape(-1, 1, 1)
        x = repeat(bottom_features, "c h w -> (n c) h w", n=num_masks) - padded_anchors
        x = functional.conv2d(x.unsqueeze(0), w1, b1, groups=num_masks).relu()
        x = functional.conv2d(x, w2, b2, groups=num_masks).relu()
        return functional.conv2d(x, w3, b3, groups=num_masks)

    def training_step(
        self,
        inputs: List[Tensor],
        classes: List[Tensor],
        masks: List[Tensor],
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        boxes = [masks_to_boxes(sample_masks) for sample_masks in masks]
        scores, reg_features = self.get_features(inputs)
        controllers = self.get_controllers(reg_features)
        pred_boxes = self.get_boxes(reg_features)
        bottom_feats = self.get_bottom_features(inputs)
        anchors = self.get_anchors(inputs, normalized=False)
        class_target, box_target, assignment = self.get_targets(
            anchors, scores, pred_boxes, classes, boxes, is_validating
        )
        class_loss, box_loss = self.get_losses(
            scores, pred_boxes, class_target, box_target
        )
        mask_losses = []  # dice loss
        batch_size = scores.shape[0]
        normed_anchors = self.get_anchors(inputs, normalized=True)
        pos_mask = reduce(class_target, "b i s -> b i", "max") == 1.0
        for batch_idx in range(batch_size):
            pos_idxs = pos_mask[batch_idx]
            if masks[batch_idx].numel() == 0 or not torch.any(pos_idxs):
                continue
            pred_masks = self.get_masks(
                params=controllers[batch_idx, pos_idxs],
                anchors=normed_anchors[pos_idxs],
                bottom_features=bottom_feats[batch_idx],
                num_masks=pos_idxs.sum(),
            ).sigmoid()
            target_masks = masks[batch_idx][assignment[batch_idx, pos_idxs]]
            mask_size = (target_masks.shape[1], target_masks.shape[2])  # FIXME ?
            pred_masks = interpolate(pred_masks, size=mask_size).squeeze(0)
            numerator = reduce(pred_masks * target_masks, "c h w -> c", "sum")
            denominator = reduce(pred_masks**2 + target_masks**2, "c h w -> c", "sum")
            mask_losses.append((1 - 2 * numerator / denominator).nan_to_num(0).mean())
        mask_loss = torch.stack(mask_losses).mean()
        return class_loss + box_loss + mask_loss, {
            "class_loss": class_loss,
            "box_loss": box_loss,
            "mask_loss": mask_loss,
        }

    def validation_step(
        self, inputs: List[Tensor], classes: List[Tensor], masks: List[Tensor]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        loss, metrics = self.training_step(inputs, classes, masks, is_validating=True)
        self.loss_computer.to(loss.device).update(loss)
        boxes = [masks_to_boxes(sample_masks) for sample_masks in masks]
        num_instances, scores, pred_classes, pred_masks = self.forward(inputs)
        map_computer_preds = []
        for batch_idx, sample_masks in enumerate(pred_masks):
            non_empty_idxs = torch.any(sample_masks > 0.5, dim=(1, 2))
            sample_boxes = masks_to_boxes(sample_masks[non_empty_idxs] > 0.5)
            map_computer_preds.append(
                {
                    "scores": scores[batch_idx][non_empty_idxs],
                    "labels": pred_classes[batch_idx][non_empty_idxs],
                    "boxes": sample_boxes,
                }
            )
        self.map_computer.to(scores.device).update(
            map_computer_preds,
            [{"labels": c.to(torch.int64), "boxes": b} for c, b in zip(classes, boxes)],
        )
        return loss, metrics

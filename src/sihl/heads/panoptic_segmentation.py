from typing import Any, Tuple, List, Dict

from einops import rearrange, repeat, reduce
from torch import nn, Tensor
from torch.nn import functional
from torchmetrics import JaccardIndex, Accuracy
from torchvision.ops import masks_to_boxes
import torch

from sihl.layers import ConvNormAct, SimpleUpscaler
from sihl.utils import interpolate

from .instance_segmentation import InstanceSegmentation


class PanopticSegmentation(InstanceSegmentation):
    """Panoptic segmentation assigns each pixel of the input to a category and an
    instance index, essentially merging instance segmentation and semantic segmentation
    tasks in one (although, in this case, instances cannot overlap).

    Refs:
        1. [CondInst](https://arxiv.org/abs/2102.03026)
    """

    def __init__(
        self,
        in_channels: List[int],
        num_stuff_classes: int,
        num_thing_classes: int,
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
        ignore_index: int = -100,
    ) -> None:
        """
        Args:
            in_channels (List[int]): Number of channels in input feature maps, sorted by level.
            num_stuff_classes (int): Number of possible stuff categories.
            num_thing_classes (int): Number of possible thing categories.
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
        """
        super().__init__(
            in_channels=in_channels,
            num_classes=num_thing_classes,
            mask_top_level=mask_top_level,
            soft_label_decay_steps=soft_label_decay_steps,
            num_channels=num_channels,
            num_layers=num_layers,
            bottom_level=bottom_level,
            top_level=top_level,
            max_instances=max_instances,
            t_min=t_min,
            t_max=t_max,
            topk=topk,
        )

        self.ignore_index = ignore_index  # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        self.num_stuff_classes = num_stuff_classes
        self.num_thing_classes = num_thing_classes
        self.semantic_branch = nn.ModuleList(
            [ConvNormAct(in_channels[bottom_level], self.num_channels)]
            + [
                nn.Sequential(
                    ConvNormAct(in_channels[level], self.num_channels),
                    *[
                        SimpleUpscaler(self.num_channels, self.num_channels)
                        for _ in range(level - bottom_level)
                    ],
                )
                for level in range(bottom_level + 1, self.mask_top_level + 1)
            ]
        )
        self.semantic_out_conv = nn.Conv2d(self.num_channels, num_stuff_classes, 1)
        self.output_shapes = {"panoptic_maps": ("batch_size", 2, "height", "width")}

    def get_semantic_map(self, inputs: List[Tensor]) -> Tensor:
        xs = inputs[self.bottom_level : self.mask_top_level + 1]
        outs = [conv(x) for x, conv in zip(xs, self.semantic_branch)]
        return self.semantic_out_conv(torch.stack(outs).sum(dim=0))

    def forward(self, inputs: List[Tensor]) -> Tensor:
        num_instances, scores, classes, masks = super().forward(inputs)
        things = self.instance_to_panoptic(scores, classes, masks)
        stuff = interpolate(
            self.get_semantic_map(inputs), size=inputs[0].shape[2:]
        ).argmax(dim=1, keepdim=True)
        stuff = stuff * (things[:, 0:1] == 0)  # zero-out thing pixels
        return things + stuff  # stuff ids are just their categories

    def panoptic_to_instance(self, maps: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        instance_classes, instance_masks = [], []
        for panoptic_map in maps:
            semantic_map, id_map = panoptic_map
            sample_instance_masks, sample_instance_classes = [], []
            for instance_id in id_map.unique():
                instance_mask = id_map == instance_id
                category_idx = semantic_map[instance_mask][0]
                if category_idx == self.ignore_index:
                    continue
                if category_idx < self.num_stuff_classes:
                    continue
                sample_instance_classes.append(category_idx - self.num_stuff_classes)
                sample_instance_masks.append(instance_mask)
            if len(sample_instance_masks):
                instance_classes.append(torch.stack(sample_instance_classes))
                instance_masks.append(torch.stack(sample_instance_masks))
            else:
                instance_classes.append(torch.empty((0,), dtype=torch.int64))
                instance_masks.append(torch.empty((0, 1, 1), dtype=torch.bool))
        return instance_classes, instance_masks

    def instance_to_panoptic(
        self, scores: Tensor, classes: Tensor, masks: Tensor
    ) -> Tensor:
        # remove low scoring pixels and low scoring instances
        masks = masks * ((masks > 0.5) & rearrange(scores > 0.5, "b i -> b i 1 1"))
        # remove overlaps by keeping the highest scoring instance only at each pixel
        masks = (masks == reduce(masks, "b i h w -> b 1 h w", "max")) & (masks > 0)
        classes = rearrange(classes + self.num_stuff_classes, "b i -> b i 1 1")
        thing_classes = reduce(masks * classes, "b i h w -> b 1 h w", "max")
        batch_size = masks.shape[0]
        all_ids = repeat(torch.arange(self.max_instances), "i -> b i 1 1", b=batch_size)
        all_ids = (all_ids + self.num_stuff_classes).to(masks.device)
        thing_ids = reduce(masks.to(all_ids) * all_ids, "b i h w -> b 1 h w", "max")
        return torch.cat([thing_classes, thing_ids], dim=1)

    def training_step(
        self,
        inputs: List[Tensor],
        targets: Tensor,
        is_validating: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        # NOTE: targets are concatenated [class_map, id_map]
        # NOTE: category indices are stuff first, then things
        instance_classes, instance_masks = self.panoptic_to_instance(targets)
        instance_loss, instance_metrics = super().training_step(
            inputs, instance_classes, instance_masks, is_validating
        )

        pred_semantic_map = interpolate(
            self.get_semantic_map(inputs), size=targets.shape[2:]
        )
        semantic_target = targets[:, 0]
        is_stuff = semantic_target < self.num_stuff_classes
        ignored = torch.full_like(semantic_target, self.ignore_index) * (~is_stuff)
        semantic_loss = functional.cross_entropy(
            pred_semantic_map,
            semantic_target * is_stuff + ignored,  # ignore thing pixels
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        metrics = dict(**instance_metrics, semantic_loss=semantic_loss)
        return instance_loss + semantic_loss, metrics

    def on_validation_start(self) -> None:
        super().on_validation_start()
        metric_kwargs = {
            "task": "multiclass",
            "num_classes": self.num_stuff_classes + self.num_thing_classes,
            "ignore_index": self.ignore_index,
        }
        self.pixel_accuracy = Accuracy(**metric_kwargs)
        self.mean_iou_computer = JaccardIndex(**metric_kwargs)

    def validation_step(
        self, inputs: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, Any]]:
        num_instances, scores, classes, masks = super().forward(inputs)
        things = self.instance_to_panoptic(scores, classes, masks)
        stuff = interpolate(
            self.get_semantic_map(inputs), size=inputs[0].shape[2:]
        ).argmax(dim=1)
        stuff = stuff * (things[:, 0] == 0)  # zero-out thing pixels

        pred_semantic = stuff + things[:, 0, :, :]
        target_semantic = targets[:, 0, :, :]
        self.mean_iou_computer.to(stuff.device).update(pred_semantic, target_semantic)
        self.pixel_accuracy.to(stuff.device).update(pred_semantic, target_semantic)

        map_computer_preds = []
        for batch_idx, sample_masks in enumerate(masks):
            non_empty_idxs = torch.any(sample_masks > 0.5, dim=(1, 2))
            sample_boxes = masks_to_boxes(sample_masks[non_empty_idxs] > 0.5)
            map_computer_preds.append(
                {
                    "scores": scores[batch_idx][non_empty_idxs],
                    "labels": classes[batch_idx][non_empty_idxs],
                    "boxes": sample_boxes,
                }
            )
        target_classes, target_masks = self.panoptic_to_instance(targets)
        target_boxes = [masks_to_boxes(_) for _ in target_masks]
        self.map_computer.to(stuff.device).update(
            map_computer_preds,
            [
                {"labels": c.to(torch.int64), "boxes": b}
                for c, b in zip(target_classes, target_boxes)
            ],
        )

        loss, metrics = self.training_step(inputs, targets, is_validating=True)
        self.loss_computer.to(loss.device).update(loss)
        return loss, metrics

    def on_validation_end(self) -> Dict[str, float]:
        metrics = super().on_validation_end()
        metrics["pixel_accuracy"] = self.pixel_accuracy.compute().item()
        metrics["mean_iou"] = self.mean_iou_computer.compute().item()
        return metrics

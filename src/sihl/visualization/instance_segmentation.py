from typing import List

from matplotlib import patches
from matplotlib import pyplot as plt
from torchvision.ops import masks_to_boxes
from torchvision.transforms.v2 import Resize
import numpy as np
import torch

from sihl.heads import InstanceSegmentation
from sihl.utils import edges

from .common import get_images, plot_to_numpy, COLORS


@get_images.register(InstanceSegmentation)
def _(head, config, input, target, features) -> List[np.ndarray]:
    # # remove empty masks
    # for idx in range(len(target["masks"])):
    #     if target["masks"][idx].shape[0] != 0:
    #         keep = target["masks"][idx].sum(dim=(1, 2)).to(torch.bool)
    #         target["masks"][idx] = target["masks"][idx][keep]
    #         target["classes"][idx] = target["classes"][idx][keep]
    boxes = [masks_to_boxes(sample_masks) for sample_masks in target["masks"]]

    categories = config["categories"] if "categories" in config else None
    prediction = head(features)
    if prediction is not None:
        num_instances, pred_scores, pred_classes, pred_masks = prediction
    resize = Resize(target["masks"][0].shape[1:], antialias=False)
    images = (resize(input).permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()
    visualizations = []
    for batch, image in enumerate(images):
        seen_categories = []
        fig, axes = plt.subplots(1, 3, figsize=(10.24, 5.12), dpi=100)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0].title.set_text("Input")
        axes[0].axis("off")
        axes[0].imshow(image)

        # sort masks by size -> draw large masks first
        axes[1].title.set_text("Target")
        if target is not None:
            for label, mask, box in zip(
                target["classes"][batch], target["masks"][batch], boxes[batch]
            ):
                category = str(label) if categories is None else categories[label]
                if category not in seen_categories:
                    seen_categories.append(category)
                color = COLORS[seen_categories.index(category) % len(COLORS)]
                axes[1].imshow(get_mask_img(mask, category, color))
                # axes[1].add_patch(get_patch(label.to("cpu"), box.to("cpu")))
        axes[2].title.set_text("Prediction")
        axes[2].imshow(np.full_like(image, fill_value=255))
        if prediction is not None:
            n = max(5, num_instances[batch])
            for label, mask in zip(pred_classes[batch, :n], pred_masks[batch, :n]):
                category = str(label) if categories is None else categories[label]
                legend = None
                if category not in seen_categories:
                    seen_categories.append(category)
                    legend = category
                color = COLORS[seen_categories.index(category) % len(COLORS)]
                axes[2].imshow(get_mask_img(mask, category, color))
                try:
                    axes[2].add_patch(
                        get_patch(
                            masks_to_boxes(mask.unsqueeze(0)).to("cpu"),
                            label.to("cpu"),
                            color,
                            legend,
                        )
                    )
                except Exception:
                    pass

        if len(seen_categories) > 0:
            fig.legend(
                handles=[
                    patches.Patch(
                        color=[_ / 255 for _ in COLORS[i % len(COLORS)] + (128,)],
                        label=label,
                    )
                    for i, label in enumerate(seen_categories)
                ],
                loc="lower center",
                frameon=False,
                ncol=min(7, len(seen_categories)),
            )
        fig.tight_layout()
        visualizations.append(plot_to_numpy(fig))
        plt.close()
    return visualizations


def get_mask_img(mask, category, color):
    mask_img = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask = (mask > 0.5).to(torch.float).to("cpu")
    edge_mask = edges(mask[None, None, :, :])[0, 0] > 0.5
    mask[edge_mask] = 1
    mask_img[..., 0] = mask * color[0]
    mask_img[..., 1] = mask * color[1]
    mask_img[..., 2] = mask * color[2]
    mask_img[..., 3] = mask * 128
    mask_img[..., 3][edge_mask] = 255
    return mask_img


def get_patch(label, box, color, legend):
    return patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=1,
        edgecolor=[_ / 255 for _ in color],
        facecolor="none",
        label=legend,
    )

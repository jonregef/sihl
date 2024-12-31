from typing import List

from matplotlib import patches
from matplotlib import pyplot as plt
from torch.nn import functional
import numpy as np
import torch

from sihl.heads import QuadrilateralDetection

from .common import get_images, plot_to_numpy, COLORS


@get_images.register(QuadrilateralDetection)
def _(
    head: QuadrilateralDetection, config, input, target, features
) -> List[np.ndarray]:
    categories = config["categories"] if "categories" in config else None
    prediction = head(features)
    if prediction is not None:
        num_instances, scores, pred_labels, pred_boxes = prediction
        saliency = head.get_saliency(features)
        saliency = functional.interpolate(
            saliency.unsqueeze(1), size=input.shape[2:], mode="nearest-exact"
        )
        saliency = saliency.squeeze(1).to("cpu").numpy()
    images = (input.permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()
    visualizations = []
    for batch_idx, image in enumerate(images):
        seen_categories = []
        fig, axes = plt.subplots(1, 3, figsize=(10, 5), dpi=100)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0].title.set_text("Input")
        axes[0].axis("off")
        axes[0].imshow(image)

        def get_patch(label, quad):
            label = str(label) if categories is None else categories[label]
            if label not in seen_categories:
                seen_categories.append(label)
                legend = label
            else:
                legend = None
            return patches.Polygon(
                quad,  # (4, 2)
                linewidth=1,
                edgecolor=[
                    _ / 255 for _ in COLORS[seen_categories.index(label) % len(COLORS)]
                ],
                facecolor="none",
                label=legend,
            )

        axes[1].title.set_text("Target")
        axes[1].imshow(np.full_like(image, fill_value=255))
        if target is not None:
            for label, quad in zip(
                target["classes"][batch_idx], target["quads"][batch_idx]
            ):
                axes[1].add_patch(get_patch(label.to("cpu"), quad.to("cpu")))
        axes[2].title.set_text("Prediction")
        axes[2].imshow(saliency[batch_idx], vmin=0, vmax=1)
        # np.full_like(image, fill_value=255))
        if prediction is not None:
            n = num_instances[batch_idx]
            for label, box in zip(
                pred_labels[batch_idx, :n], pred_boxes[batch_idx, :n]
            ):
                axes[2].add_patch(get_patch(label.to("cpu"), box.to("cpu")))
        fig.legend(loc="lower center", frameon=False, ncol=min(7, len(seen_categories)))
        fig.tight_layout()
        visualizations.append(plot_to_numpy(fig))
        plt.close()
    return visualizations

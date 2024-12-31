from typing import List

from matplotlib import pyplot as plt
import numpy as np
import torch

from sihl.heads import MulticlassClassification

from .common import get_images, plot_to_numpy


@get_images.register(MulticlassClassification)
def _(head, config, input, target, features) -> List[np.ndarray]:
    categories = config["categories"] if "categories" in config else None
    prediction = head(features)
    images = (input.permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()
    visualizations = []
    for batch, image in enumerate(images):
        fig, axes = plt.subplots(1, 3, figsize=(10.24, 5.12), dpi=100)
        for ax in axes:
            ax.axis("off")
        axes[0].title.set_text("Input")
        axes[0].imshow(image)

        def set_text(ax, label):
            label = str(label) if categories is None else categories[label]
            ax.text(
                0.5,
                0.5,
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize="large",
                bbox=dict(edgecolor="black", facecolor="white", boxstyle="round"),
            )

        axes[1].title.set_text("Target")
        axes[1].imshow(np.full_like(image, fill_value=255))
        if target is not None:
            set_text(axes[1], target[batch])
        axes[2].title.set_text("Prediction")
        axes[2].imshow(np.full_like(image, fill_value=255))
        if prediction is not None:
            set_text(axes[2], prediction[1][batch])
        fig.tight_layout()
        visualizations.append(plot_to_numpy(fig))
        plt.close()
    return visualizations

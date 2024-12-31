from typing import List

from matplotlib import pyplot as plt
import numpy as np
import torch

from sihl.heads import MultilabelClassification

from .common import get_images, plot_to_numpy


@get_images.register(MultilabelClassification)
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

        def set_text(ax, text):
            ax.text(
                0.5,
                0.5,
                text,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize="large",
                bbox=dict(edgecolor="black", facecolor="white", boxstyle="round"),
            )

        axes[1].title.set_text("Target")
        axes[1].imshow(np.full_like(image, fill_value=255))
        if target is not None:
            labels = target[batch].to(torch.bool)
            patch_text = ""
            for idx, is_true in enumerate(labels):
                if is_true:
                    patch_text += str(idx) if categories is None else categories[idx]
                    patch_text += "\n"
            set_text(axes[1], patch_text.strip())
        axes[2].title.set_text("Prediction")
        axes[2].imshow(np.full_like(image, fill_value=255))
        if prediction is not None:
            scores = prediction[0][batch].cpu().numpy()
            labels = prediction[1][batch].cpu().numpy()
            patch_text = ""
            for score, label in zip(scores, labels):
                if score < 0.5:
                    continue
                patch_text += str(label) if categories is None else categories[label]
                patch_text += "\n"
            set_text(axes[2], patch_text.strip())
        # fig.tight_layout()
        visualizations.append(plot_to_numpy(fig))
        plt.close()
    return visualizations

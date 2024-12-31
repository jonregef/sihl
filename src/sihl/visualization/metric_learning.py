from typing import List

from matplotlib import pyplot as plt
from torch.nn import functional
import numpy as np
import torch

from sihl.heads import MetricLearning

from .common import get_images, plot_to_numpy


@get_images.register(MetricLearning)
def _(head, config, input, target, features) -> List[np.ndarray]:
    embeddings = head(features)
    similarities = functional.linear(embeddings, head.index_embeddings)
    top5_values, top5_labels = torch.topk(similarities, k=6)
    top5_labels = head.index_ids[top5_labels][:, 1:6].cpu().numpy()  # argmax?
    top5_scores = top5_values[:, 1:6].cpu().numpy()
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
            )

        axes[1].title.set_text("Target")
        axes[1].imshow(np.full_like(image, fill_value=255))
        if target is not None:
            set_text(axes[1], "id: " + str(target[batch].item()))
        axes[2].title.set_text("Prediction")
        axes[2].imshow(np.full_like(image, fill_value=255))
        if embeddings is not None:
            patch_text = ""
            for rank, (score, label) in enumerate(
                zip(top5_scores[batch], top5_labels[batch])
            ):
                patch_text += (
                    f"{rank+1}. id: {str(label)} ({int(score*100)} % similar)\n"
                )
            set_text(axes[2], patch_text.strip())
        fig.tight_layout()
        visualizations.append(plot_to_numpy(fig))
        plt.close()
    return visualizations

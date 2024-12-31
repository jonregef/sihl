from typing import List

from matplotlib import pyplot as plt
import numpy as np
import torch

from sihl.heads import Autoencoding

from .common import get_images, plot_to_numpy


@get_images.register(Autoencoding)
def _(head, config, input, target, features) -> List[np.ndarray]:
    reconstructions, embeddings = head(features)
    reconstructions = (
        (reconstructions.permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()
    )
    images = (input.permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()
    target = (target.permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()
    visualizations = []
    for batch, image in enumerate(images):
        fig, axes = plt.subplots(1, 3, figsize=(10.24, 5.12), dpi=100)
        for ax in axes:
            ax.axis("off")
        axes[0].title.set_text("Input")
        axes[0].imshow(image)
        axes[1].title.set_text("Target")
        axes[1].imshow(target[batch])
        axes[2].title.set_text("Prediction")
        axes[2].imshow(reconstructions[batch])
        fig.tight_layout()
        visualizations.append(plot_to_numpy(fig))
        plt.close()
    return visualizations

from typing import List

from matplotlib import pyplot as plt
import numpy as np
import torch

from sihl.heads import ViewInvarianceLearning

from .common import get_images, plot_to_numpy


@get_images.register(ViewInvarianceLearning)
def _(head, config, input, target, features) -> List[np.ndarray]:
    cross_correlation = head.get_correlation(features, target)
    target = torch.eye(cross_correlation.shape[0], device=cross_correlation.device)
    cross_correlation = (1 - cross_correlation.abs()).to("cpu").numpy()
    target = (1 - target).to("cpu").numpy()
    image = (input.permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()[0]
    visualizations = []
    fig, axes = plt.subplots(1, 3, figsize=(10.24, 5.12), dpi=100)
    for ax in axes:
        ax.axis("off")
    axes[0].title.set_text("Input")
    axes[0].imshow(image)
    axes[1].title.set_text("Target")
    axes[1].imshow(target, cmap="gray")
    axes[2].title.set_text("Prediction")
    axes[2].imshow(cross_correlation, cmap="gray", vmin=0, vmax=1)
    fig.tight_layout()
    visualizations.append(plot_to_numpy(fig))
    plt.close()
    return visualizations

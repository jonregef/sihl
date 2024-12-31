from typing import List

from matplotlib import pyplot as plt
from torchvision.transforms.v2 import Resize
import numpy as np
import torch

from sihl.heads import DepthEstimation

from .common import get_images, plot_to_numpy


@get_images.register(DepthEstimation)
def _(head, config, input, target, features) -> List[np.ndarray]:
    prediction = head(features)
    target = target["targets"]  # TODO: masks
    resize = Resize(target[0].shape[1:], antialias=False)
    images = (resize(input).permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()
    visualizations = []
    for batch, image in enumerate(images):
        fig, axes = plt.subplots(1, 3, figsize=(10.24, 5.12), dpi=100)
        for ax in axes:
            ax.axis("off")
        axes[0].title.set_text("Input")
        axes[0].imshow(image)
        axes[1].title.set_text("Target")
        target_depth_map = target[batch].to("cpu")
        alpha = np.ones_like(target_depth_map)
        if "ignore" in config:
            alpha[target_depth_map == config["ignore"]] = 0
        axes[1].imshow(
            target_depth_map,
            cmap="Greys",
            vmin=target_depth_map.min(),
            vmax=target_depth_map.max(),
            alpha=alpha,
        )
        axes[2].title.set_text("Prediction")
        axes[2].imshow(
            prediction[batch].to("cpu"),
            cmap="Greys",
            vmin=target_depth_map.min(),
            vmax=target_depth_map.max(),
            alpha=alpha,
        )
        fig.tight_layout()
        visualizations.append(plot_to_numpy(fig))
        plt.close()
    return visualizations

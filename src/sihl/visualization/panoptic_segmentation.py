from typing import List

from matplotlib import patches
from matplotlib import pyplot as plt
from torchvision.transforms.v2 import Resize
import numpy as np
import torch

from sihl.heads import PanopticSegmentation

from .common import get_images, plot_to_numpy, COLORS


@get_images.register(PanopticSegmentation)
def _(head, config, input, target, features) -> List[np.ndarray]:
    categories = config["categories"] if "categories" in config else None
    prediction = head.forward(features)[:, 0, :, :]  # semantic map
    target = target[:, 0, :, :]  # semantic map
    resize = Resize(target[0].shape[1:], antialias=False)
    images = (resize(input).permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()

    visualizations = []
    for batch, image in enumerate(images):
        seen_categories = []
        fig, axes = plt.subplots(1, 3, figsize=(10, 5), dpi=100)
        for ax in axes:
            ax.axis("off")
        axes[0].title.set_text("Input")
        axes[0].imshow(image)

        def get_map_img(pixelmap):
            shape = (pixelmap.shape[0], pixelmap.shape[1], 4)
            map_img = np.full(shape, fill_value=255, dtype=np.uint8)
            for idx in pixelmap.unique():
                if idx == config.get("ignore", None):
                    continue
                label = str(idx) if categories is None else categories[idx]
                if label not in seen_categories:
                    seen_categories.append(label)
                color = COLORS[seen_categories.index(label) % len(COLORS)]
                mask = (pixelmap == idx).to(torch.uint8)
                map_img[..., 0] = mask * color[0] + (1 - mask) * map_img[..., 0]
                map_img[..., 1] = mask * color[1] + (1 - mask) * map_img[..., 1]
                map_img[..., 2] = mask * color[2] + (1 - mask) * map_img[..., 2]
                map_img[..., 3] = mask * 255 + (1 - mask) * map_img[..., 3]
            return map_img

        axes[1].title.set_text("Target")
        if target is not None:
            axes[1].imshow(get_map_img(target[batch].to("cpu")))
        axes[2].title.set_text("Prediction")
        if prediction is not None:
            axes[2].imshow(get_map_img(prediction[batch].to("cpu")))

        fig.legend(
            handles=[
                patches.Patch(
                    color=[_ / 255 for _ in COLORS[i % len(COLORS)]], label=label
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

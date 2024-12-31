from typing import List

from matplotlib import pyplot as plt
import numpy as np
import torch

from sihl.heads import AutoregressiveTextRecognition

from .common import get_images, plot_to_numpy


@get_images.register(AutoregressiveTextRecognition)
def _(head, config, input, target, features) -> List[np.ndarray]:
    prediction = head(features)
    if "tokens" in config:
        tokens = config["tokens"]
    else:
        tokens = [str(_) for _ in range(head.num_tokens)]

    if isinstance(tokens, str):
        tokens = list(tokens)

    tokens = tokens + [""]

    images = (input.permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()
    visualizations = []

    for batch, image in enumerate(images):
        blank_image = np.full_like(image, fill_value=255)

        fig, axes = plt.subplots(1, 3, figsize=(10.24, 5.12), dpi=100)
        for ax in axes:
            ax.axis("off")
        axes[0].title.set_text("Input")
        axes[0].imshow(image)

        axes[1].title.set_text("Target")
        axes[1].imshow(blank_image)
        if target is not None:
            patch_text = " ".join(
                [tokens[idx] if tokens else f"<{idx}>" for idx in target[batch]]
            )
            set_text(axes[1], patch_text.strip() + "\n\n")
        axes[2].title.set_text("Prediction")
        axes[2].imshow(blank_image)

        pred_tokens = prediction[batch]  # [:num_chars]
        patch_text = " ".join([tokens[idx] for idx in pred_tokens]).strip()
        set_text(axes[2], patch_text)

        fig.tight_layout()
        visualizations.append(plot_to_numpy(fig))
        plt.close()
    return visualizations


def set_text(ax, text):
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize="large",
        wrap=True,
    )


def set_text_on_peak(ax, image, text, fontsize=8):
    peak = np.unravel_index(np.argmax(image), image.shape)
    ax.text(
        peak[1],
        peak[0],
        text,
        fontsize=fontsize,
        color="black",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5},
    )

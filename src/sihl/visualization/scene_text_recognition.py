from typing import List

from matplotlib import pyplot as plt
import numpy as np
import torch

from sihl.heads import SceneTextRecognition

from .common import get_images, plot_to_numpy


RAINBOW_CMAPS = ["Reds", "Oranges", "Greens", "Blues", "Purples"]


@get_images.register(SceneTextRecognition)
def _(head, config, input, target, features) -> List[np.ndarray]:
    prediction = head(features)
    if "tokens" in config:
        tokens = config["tokens"]
    else:
        tokens = [str(_) for _ in range(head.num_tokens)]

    if isinstance(tokens, str):
        tokens = list(tokens)
    images = (input.permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()
    index_map, token_map = head.get_maps(features)
    index_map, token_map = index_map.to("cpu").numpy(), token_map.to("cpu").numpy()
    visualizations = []

    for batch, image in enumerate(images):
        blank_image = np.full_like(image, fill_value=255)

        fig, axes = plt.subplots(2, 3, figsize=(10.24, 5.12), dpi=100)
        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")
        axes[0][0].title.set_text("Input")
        axes[0][0].imshow(image)

        axes[0][1].title.set_text("Target")
        axes[0][1].imshow(blank_image)
        if target is not None:
            patch_text = " ".join(
                [tokens[idx] if tokens else f"<{idx}>" for idx in target[batch]]
            )
            set_text(axes[0][1], patch_text.strip() + "\n\n")
        axes[0][2].title.set_text("Prediction")
        axes[0][2].imshow(blank_image)
        if prediction is not None:
            scores = prediction[0][batch].cpu().numpy()
            positive_indices = np.argwhere(scores > 0.5).flatten().tolist()
            if len(positive_indices):
                num_chars = positive_indices[-1] + 1
                pred_tokens = prediction[1][batch][:num_chars]
                scores = scores[:num_chars]

                axes[1][0].title.set_text("Index map")
                for idx, map_slice in enumerate(index_map[batch][:num_chars]):
                    axes[1][0].imshow(
                        map_slice,
                        cmap=RAINBOW_CMAPS[idx % len(RAINBOW_CMAPS)],
                        vmin=0,
                        vmax=1,
                        alpha=map_slice * map_slice.max(),
                        interpolation="none",
                    )
                    set_text_on_peak(axes[1][0], map_slice, text=str(idx))

                axes[1][1].title.set_text("Token map")
                for idx, map_slice in enumerate(token_map[batch]):
                    if idx not in pred_tokens or idx >= len(tokens):
                        continue
                    axes[1][1].imshow(
                        map_slice,
                        cmap=RAINBOW_CMAPS[idx % len(RAINBOW_CMAPS)],
                        vmin=0,
                        vmax=1,
                        alpha=map_slice * map_slice.max(),
                        interpolation="none",
                    )
                    set_text_on_peak(axes[1][1], map_slice, text=tokens[idx])

                patch_text = " ".join(
                    [tokens[idx] for idx in pred_tokens if idx < len(tokens)]
                ).strip()
                patch_text += "\nscores: " + str([round(s, 2) for s in scores.tolist()])
            else:
                patch_text = f"\nmax score: {scores.max()}"

            set_text(axes[0][2], patch_text)

        axes[1][2].imshow(blank_image)
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

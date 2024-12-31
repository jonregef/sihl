from typing import List

from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor
import numpy as np
import torch

from sihl.heads import AnomalyDetection
from sihl.utils import interpolate

from .common import get_images, plot_to_numpy


def pca_reduce(x: Tensor, k: int) -> Tensor:
    N, C, H, W = x.shape
    x = x.view(N, C, -1)
    x_centered = x - x.mean(dim=(0, 2), keepdim=True)
    U, S, V = torch.svd(x_centered.transpose(1, 2))
    return (U[:, :, :k] @ torch.diag_embed(S[:, :k])).transpose(1, 2).view(N, k, H, W)


def normalized(x: Tensor) -> Tensor:
    return (x - x.min()) / (x.max() - x.min())


def to_rgb(x):
    return normalized(pca_reduce(x.unsqueeze(0), 3)[0].permute(1, 2, 0))


@get_images.register(AnomalyDetection)
def _(head, config, input, target, features) -> List[np.ndarray]:
    prediction = head(features)
    image = features[0]
    teacher_out = ((features[head.level] - head.features_mean) / head.feature_std).to(
        "cpu"
    )
    student_out = head.student(image)
    st_teacher = student_out[:, : head.out_channels].to("cpu")
    st_autoencoder = student_out[:, head.out_channels :].to("cpu")

    encoded = head.autoencoder_encoder(image)
    old_size = encoded.shape[2:]
    encoded = head.autoencoder_resize(encoded)
    h, w = encoded.shape[2:]
    encoded = rearrange(encoded, "b c h w -> b (c h w)")
    encoded = rearrange(
        head.autoencoder_bottleneck(encoded), "b (c h w) -> b c h w", h=h, w=w
    )
    ae_output = head.autoencoder_decoder(interpolate(encoded, size=old_size)).to("cpu")

    distance_st, distance_ae, distance_stae = head.compute_distances(features)
    local_anomaly = distance_st.mean(dim=1)
    global_anomaly = distance_stae.mean(dim=1)
    local_anomaly = (
        head.local_thresh
        * (local_anomaly - head.q_st_start)
        / (head.q_st_end - head.q_st_start)
    ).to("cpu")
    global_anomaly = (
        head.global_thresh
        * (global_anomaly - head.q_ae_start)
        / (head.q_ae_end - head.q_ae_start)
    ).to("cpu")

    images = (input.permute(0, 2, 3, 1) * 255).to(torch.uint8).to("cpu").numpy()
    visualizations = []

    for batch, image in enumerate(images):
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), dpi=100)
        for ax in axes:
            for subax in ax:
                subax.axis("off")
        axes[0][0].title.set_text("Input")
        axes[0][0].imshow(image)
        axes[0][1].title.set_text("Target")
        if target is not None:
            axes[0][1].imshow(target[batch].to("cpu"), cmap="gray")
        axes[0][2].title.set_text("Prediction")
        if prediction is not None:
            axes[0][2].imshow(prediction[batch].to("cpu"), cmap="gray", vmin=0, vmax=1)

        axes[1][0].title.set_text("autoencoder")
        axes[1][0].imshow(to_rgb(ae_output[batch]))
        axes[1][1].title.set_text("student (autoencoder)")
        axes[1][1].imshow(to_rgb(st_autoencoder[batch]))
        axes[1][2].title.set_text("Global anomaly")
        axes[1][2].imshow(global_anomaly[batch], cmap="seismic", vmin=-1, vmax=1)

        axes[2][0].title.set_text("teacher")
        axes[2][0].imshow(to_rgb(teacher_out[batch]))
        axes[2][1].imshow(to_rgb(st_teacher[batch]))
        axes[2][1].title.set_text("student (teacher)")
        axes[2][2].title.set_text("Local anomaly")
        axes[2][2].imshow(local_anomaly[batch], cmap="seismic", vmin=-1, vmax=1)

        fig.tight_layout()
        visualizations.append(plot_to_numpy(fig))
        plt.close()
    return visualizations

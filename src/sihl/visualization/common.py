from functools import singledispatch
from io import BytesIO
from typing import List

from matplotlib.pyplot import Figure
import numpy as np

COLORS = (
    (0, 0, 0),
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
    (255, 255, 255),
)


def plot_to_numpy(fig: Figure) -> np.ndarray:
    io_buf = BytesIO()
    fig.savefig(io_buf, format="raw", dpi=100)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )[:, :, :3]
    io_buf.close()
    return np.moveaxis(img_arr, 2, 0)  # HWC -> CHW


@singledispatch
def get_images(head, config, input, target, features) -> List[np.ndarray]:
    raise NotImplementedError(f'got unknown type "{type(head)}"')

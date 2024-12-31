import os
import pytest
import tempfile

import torch

from sihl import TimmBackbone, TIMM_BACKBONE_NAMES


@pytest.mark.parametrize("name", TIMM_BACKBONE_NAMES)
def test_torchvision_backbones(name: str) -> None:
    image = torch.rand((4, 3, 32, 32))
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ["HF_HUB_CACHE"] = tmpdirname
        model = TimmBackbone(name)
    assert len(model.out_channels) == 6
    y = model(image)
    assert [_.shape[0] for _ in y] == [4 for _ in y]
    assert [_.shape[1] for _ in y] == model.out_channels
    assert [_.shape[2] for _ in y] == [32 // 2**idx for idx in range(len(y))]
    assert [_.shape[3] for _ in y] == [32 // 2**idx for idx in range(len(y))]


def test_grayscale_input() -> None:
    image = torch.rand((4, 1, 32, 32))
    model = TimmBackbone("resnet18", input_channels=1)
    assert model.out_channels[0] == 1
    y = model(image[:, 0:1, :, :])
    assert y[0].shape[1] == 1


def test_multispectral_input() -> None:
    image = torch.rand((4, 9, 32, 32))
    model = TimmBackbone("resnet18", input_channels=9)
    assert model.out_channels[0] == 9
    y = model(image)
    assert y[0].shape[1] == 9


def test_freeze_all_layers() -> None:
    TimmBackbone("resnet18", frozen_levels=-1)


def test_freeze_some_layers() -> None:
    TimmBackbone("resnet18", frozen_levels=3)


def test_top_level() -> None:
    image = torch.rand((4, 3, 2**7, 2**7))
    model = TimmBackbone("resnet18", top_level=7)
    assert len(model.out_channels) == 8
    y = model(image)
    assert len(y) == 8
    assert y[7].shape[2] == y[7].shape[3] == 1
    assert y[6].shape[2] == y[6].shape[3] == 2


def test_unrecognized_backbone() -> None:
    with pytest.raises(ValueError):
        TimmBackbone("resnet17")

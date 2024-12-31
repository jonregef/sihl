# ruff: noqa: F401, F811
from torch import Tensor
import lightning.pytorch as pl
import pytest
import torch

from sihl import SihlLightningModule, SihlModel, TorchvisionBackbone, TimmBackbone
from sihl.heads import MulticlassClassification

from heads.test_multiclass_classification import model, targets


@pytest.fixture()
def torchvision_module(model: MulticlassClassification) -> SihlLightningModule:
    return SihlLightningModule(
        SihlModel(backbone=TorchvisionBackbone("resnet18"), neck=None, heads=[model])
    )


@pytest.fixture()
def timm_module(model: MulticlassClassification) -> SihlLightningModule:
    return SihlLightningModule(
        SihlModel(backbone=TimmBackbone("resnet18"), neck=None, heads=[model])
    )


@pytest.mark.parametrize("module", (torchvision_module, timm_module))
def test_forward_pass(module: SihlLightningModule, request) -> None:
    module = request.getfixturevalue(module.__name__)
    module(module.model.backbone.dummy_input)


@pytest.mark.parametrize("module", (torchvision_module, timm_module))
def test_training_step(module: SihlLightningModule, targets: Tensor, request) -> None:
    module = request.getfixturevalue(module.__name__)
    module.trainer = pl.Trainer()
    loss = module.training_step((torch.rand((4, 3, 32, 32)), [targets]), 0)
    assert loss.item()


@pytest.mark.parametrize("module", (torchvision_module, timm_module))
def test_validation_step(module: SihlLightningModule, targets: Tensor, request) -> None:
    module = request.getfixturevalue(module.__name__)
    module.trainer = pl.Trainer()
    module.on_validation_start()
    loss = module.validation_step((torch.rand((4, 3, 32, 32)), [targets]), 0)
    assert loss.item()
    module.on_validation_epoch_end()


@pytest.mark.parametrize("module", (torchvision_module, timm_module))
def test_configure_optimizers(module: SihlLightningModule, request) -> None:
    module = request.getfixturevalue(module.__name__)
    module.configure_optimizers()

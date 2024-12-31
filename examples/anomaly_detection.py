from pathlib import Path
from random import shuffle
from typing import Tuple
import logging

from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import tv_tensors
from torchvision.io.image import ImageReadMode
import kaggle
import lightning
import lightning.pytorch as pl
import torch
import torchvision
import torchvision.transforms.v2 as transforms

from sihl import SihlModel, SihlLightningModule, TorchvisionBackbone
from sihl.heads import AnomalyDetection


class MVTecDataset(torch.utils.data.Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, data_dir: Path, train: bool, subtask: str = "cable") -> None:
        self.train = train
        self.data_dir = data_dir / subtask
        self.images = list(
            (self.data_dir / ("train" if train else "test")).rglob("*.png")
        )
        shuffle(self.images)
        self.transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

    def __getitem__(self, idx: int) -> Tensor:
        image_path = self.images[idx]
        image = torchvision.io.read_image(
            str(image_path.resolve()), mode=ImageReadMode.RGB
        )
        if not self.train:
            gt_dir = self.data_dir / "ground_truth" / image_path.parent.name
            gt_file = gt_dir / f"{image_path.stem}_mask.png"
            if gt_file.exists():
                mask = torchvision.io.read_image(str(gt_file.resolve())).to(torch.bool)
                image, mask = self.transforms(image, tv_tensors.Mask(mask))
                return image, mask.squeeze(0)
        image = self.transforms(image)
        mask = torch.zeros(image.shape[1:], dtype=torch.bool)
        return image, mask

    def __len__(self) -> int:
        return len(self.images)


class MVTecDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, subtask: str = "leather") -> None:
        super().__init__()
        self.batch_size = batch_size
        self.subtask = subtask
        self.data_dir = Path(__file__).parent / "data" / "MVTec"

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "ipythonx/mvtec-ad", path=self.data_dir, unzip=True, quiet=False
            )

    def setup(self, stage: str = "") -> None:
        self.trainset = MVTecDataset(self.data_dir, train=True, subtask=self.subtask)
        self.valset = MVTecDataset(self.data_dir, train=False, subtask=self.subtask)

    def train_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=Path(__file__).parent / "logs", name="anomaly_detection"
    )
    lightning.seed_everything(0)

    datamodule = MVTecDataModule(batch_size=1, subtask="cable")
    datamodule.prepare_data()
    datamodule.setup()
    trainer = pl.Trainer(
        max_steps=70_000,
        accelerator="gpu",
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        precision="16-mixed",
        log_every_n_steps=len(datamodule.trainset) // datamodule.batch_size,
    )

    with trainer.init_module():
        LEVEL = 3
        backbone = TorchvisionBackbone(
            "resnet50",
            pretrained=True,
            top_level=LEVEL,
            frozen_levels=-1,
            freeze_batchnorms=True,
        )
        head = AnomalyDetection(in_channels=backbone.out_channels, level=LEVEL)
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=None, heads=[head]),
            optimizer=torch.optim.AdamW,
        )

    if hasattr(head, "pretraining_step"):
        head.eval()
        with torch.no_grad():
            model.on_pretraining_start()
            for batch in datamodule.train_dataloader():
                model.pretraining_step(batch)
            model.on_pretraining_end()
        head.train()

    log.debug(
        summary(
            model,
            row_settings=("var_names",),
            col_names=("num_params", "trainable"),
            verbose=0,
            depth=4,
        )
    )
    trainer.fit(model, datamodule=datamodule)

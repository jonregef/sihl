from pathlib import Path
from typing import Tuple, Literal

from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.io.image import ImageReadMode
import kaggle
import lightning
import lightning.pytorch as pl
import logging
import torch
import torchvision
import torchvision.transforms.v2 as transforms

from sihl import SihlModel, SihlLightningModule
from sihl.heads import Autoencoding
from sihl.torchvision_backbone import TorchvisionBackbone


class StanfordCarsDataset(torch.utils.data.Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, data_dir: Path, split: Literal["train", "test"]) -> None:
        self.split = split
        self.data_dir = data_dir
        self.images = list((data_dir / "car_data" / "car_data" / split).rglob("*.jpg"))

        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path = self.images[idx]
        image = self.transform(
            torchvision.io.read_image(str(image_path.resolve()), mode=ImageReadMode.RGB)
        )
        return image, image

    def __len__(self) -> int:
        return len(self.images)


class StanfordCarsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(__file__).parent / "data" / "StanfordCars"

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "jutrera/stanford-car-dataset-by-classes-folder",
                path=self.data_dir,
                unzip=True,
                quiet=False,
            )

    def setup(self, stage: str = "") -> None:
        self.trainset = StanfordCarsDataset(self.data_dir, split="train")
        self.validset = StanfordCarsDataset(self.data_dir, split="test")

    def train_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.validset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=Path(__file__).parent / "logs", name="autoencoding"
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        precision="16-mixed",
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone("resnet50", pretrained=True)
        head = Autoencoding(
            in_channels=backbone.out_channels,
            prebottleneck_size=(8, 8),
            representation_channels=1024,
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=None, heads=[head]),
            optimizer=torch.optim.AdamW,
        )

    log.debug(
        summary(
            model,
            row_settings=("var_names",),
            col_names=("num_params", "trainable"),
            verbose=0,
            depth=4,
        )
    )
    trainer.fit(model, datamodule=StanfordCarsDataModule(batch_size=32))

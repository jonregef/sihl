from pathlib import Path
from typing import Tuple, List

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
from sihl.heads import ViewInvarianceLearning
from sihl.torchvision_backbone import TorchvisionBackbone


class StanfordCarsDataset(torch.utils.data.Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, data_dir: Path, train: bool) -> None:
        self.data_dir = data_dir
        self.images: List[Tuple[Path, int]] = []
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(),
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

        img_dir = data_dir / "car_data" / "car_data" / ("train" if train else "test")
        self.categories = sorted([_.name for _ in img_dir.iterdir()])
        for img in img_dir.rglob("*.jpg"):
            annot = img.parent.name
            self.images.append((img, self.categories.index(annot)))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path, target = self.images[idx]
        image = torchvision.io.read_image(
            str(image_path.resolve()), mode=ImageReadMode.RGB
        )
        return self.transform(image), self.transform(image)

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
        self.trainset = StanfordCarsDataset(self.data_dir, train=True)
        self.validset = StanfordCarsDataset(self.data_dir, train=False)

    def train_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=Path(__file__).parent / "logs", name="view_invariance_learning"
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        log_every_n_steps=10,
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        precision="16-mixed",
    )

    with trainer.init_module():
        backbone = TorchvisionBackbone("resnet50", pretrained=True)
        head = ViewInvarianceLearning(
            in_channels=backbone.out_channels, embedding_dim=64
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=None, heads=[head]),
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-2},
        )

    model.visualize_n_batches = 1
    model.visualize_n_per_batch = 128
    log.info(
        summary(
            model,
            row_settings=("var_names",),
            col_names=("num_params", "trainable"),
            verbose=0,
            depth=4,
        )
    )
    trainer.fit(model, datamodule=StanfordCarsDataModule(batch_size=128))

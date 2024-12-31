from pathlib import Path
from typing import Tuple, List
import logging
import random

from lightning.pytorch.callbacks import Callback
from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.io.image import ImageReadMode
import kaggle
import lightning
import lightning.pytorch as pl
import torch
import torchvision
import torchvision.transforms.v2 as transforms

from sihl import SihlModel, SihlLightningModule, TorchvisionBackbone
from sihl.heads import MetricLearning


class QuerySetCallback(Callback):
    def __init__(self, index_set_dataloader: DataLoader) -> None:
        super().__init__()
        self.index_set_dataloader = index_set_dataloader

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        assert isinstance(pl_module.model.heads[0], MetricLearning)
        with torch.no_grad():
            pl_module.model.heads[0].reset_validation_index_set()
            for batch in self.index_set_dataloader:
                inputs = pl_module.model.extract_features(batch[0].to(pl_module.device))
                pl_module.model.heads[0].extend_validation_index_set(
                    inputs, batch[1].to(pl_module.device)
                )

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.model.heads[0].reset_validation_index_set()


class StanfordCarsDataset(torch.utils.data.Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, data_dir: Path, ids: List[str], split: str) -> None:
        self.split = split
        self.data_dir = data_dir
        self.ids = ids
        self.images: List[Tuple[Path, int]] = []
        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                    ),
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

        for img in data_dir.rglob("*.jpg"):
            annot = img.parent.name
            if annot in ids:
                self.images.append((img, self.ids.index(annot)))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path, target = self.images[idx]
        image = torchvision.io.read_image(
            str(image_path.resolve()), mode=ImageReadMode.RGB
        )
        return self.transform(image), torch.tensor(target)

    def __len__(self) -> int:
        return len(self.images)


class StanfordCarsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(__file__).parent / "data" / "StanfordCars"
        with open(self.data_dir / "names.csv") as f:
            ids = f.read().splitlines()
        ids = [_.replace("/", "-") for _ in ids]
        random.shuffle(ids)
        self.ids = ids

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
        split = len(self.ids) // 2
        self.trainset = StanfordCarsDataset(
            self.data_dir, self.ids[split:], split="train"
        )
        self.validset = StanfordCarsDataset(
            self.data_dir, self.ids[:split], split="valid"
        )

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
        save_dir=Path(__file__).parent / "logs", name="metric_learning"
    )
    lightning.seed_everything(0)

    datamodule = StanfordCarsDataModule(batch_size=128)
    datamodule.prepare_data()
    datamodule.setup()

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        logger=logger,
        callbacks=[
            pl.callbacks.RichProgressBar(leave=True),
            QuerySetCallback(datamodule.val_dataloader()),
        ],
        precision="16-mixed",
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone("resnet50", pretrained=True)
        head = MetricLearning(
            in_channels=backbone.out_channels,
            num_identities=len(datamodule.ids),
            embedding_dim=1024,
            margin=1 / len(datamodule.ids),
        )
        model = SihlLightningModule(
            model=SihlModel(backbone=backbone, neck=None, heads=[head]),
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-5},
        )

    log.info(
        summary(
            model,
            row_settings=("var_names",),
            col_names=("num_params", "trainable"),
            verbose=0,
            depth=4,
        )
    )
    trainer.fit(model, datamodule=datamodule)

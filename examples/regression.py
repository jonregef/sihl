from pathlib import Path
from typing import Tuple, List
from random import shuffle
import logging
import sys

from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary
import kaggle
import lightning
import lightning.pytorch as pl
import torch
import torchvision
import torchvision.transforms.v2 as transforms

sys.path.append("../src")


from sihl.heads import Regression
from sihl import SihlModel, SihlLightningModule
from sihl.torchvision_backbone import TorchvisionBackbone


class AgePredictionDataset(torch.utils.data.Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, data_dir: Path, train: bool = False) -> None:
        self.train = train
        self.data_dir = data_dir
        self.images_and_targets: List[Tuple[Path, float]] = []
        for subdir in data_dir.iterdir():
            age = float(subdir.name)
            for imgf in subdir.glob("*.jpg"):
                self.images_and_targets.append((imgf, age))
        shuffle(self.images_and_targets)
        if self.train:
            self.preprocess = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.preprocess = transforms.ToDtype(torch.float32, scale=True)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path, target = self.images_and_targets[idx]
        image = torchvision.io.read_image(
            str(image_path.resolve()), mode=torchvision.io.ImageReadMode.RGB
        )
        image = self.preprocess(image)
        return image, torch.tensor(target)

    def __len__(self) -> int:
        return len(self.images_and_targets)


class AgePredictionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(__file__).parent / "data" / "AgePrediction"

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "mariafrenti/age-prediction",
                path=self.data_dir,
                unzip=True,
                quiet=False,
            )

    def setup(self, stage: str = "") -> None:
        self.trainset = AgePredictionDataset(
            self.data_dir / "age_prediction_up" / "age_prediction" / "train", train=True
        )
        self.valset = AgePredictionDataset(
            self.data_dir / "age_prediction_up" / "age_prediction" / "test", train=False
        )

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
        save_dir=Path(__file__).parent / "logs", name="regression"
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        val_check_interval=0.5,
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        precision="16-mixed",
    )

    with trainer.init_module():
        backbone = TorchvisionBackbone("resnet50", pretrained=True)
        head = Regression(
            in_channels=backbone.out_channels, lower_bound=0, upper_bound=100
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=None, heads=[head]),
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-4},
        )

    log.debug(
        summary(
            model.model,
            input_data=model.model.backbone.dummy_input,
            row_settings=("var_names",),
            col_names=("output_size", "num_params", "mult_adds", "trainable"),
            verbose=0,
            depth=4,
        )
    )
    trainer.fit(model, datamodule=AgePredictionDataModule(batch_size=64))

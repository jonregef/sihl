from pathlib import Path
from typing import Tuple, List, Literal, Dict
import csv
import logging

from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary
import kaggle
import lightning
import lightning.pytorch as pl
import torch
import torchvision

from sihl import SihlModel, SihlLightningModule
from sihl.heads import TextRecognition
from sihl.torchvision_backbone import TorchvisionBackbone
from sihl.utils import random_pad


def collate_fn(batch: List[Tuple[Tensor, Dict]]):
    batch = (
        torch.stack([sample[0] for sample in batch]),
        [[sample[1] for sample in batch]],  # (num_heads, num_samples)
    )
    return batch


class CyrillicDataset(torch.utils.data.Dataset[Tuple[Tensor, bool]]):
    max_length = 40
    tokens = [
        " ",
        "!",
        '"',
        "%",
        "'",
        "(",
        ")",
        "+",
        ",",
        "-",
        ".",
        "/",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ":",
        ";",
        "=",
        "?",
        "R",
        "[",
        "]",
        "b",
        "c",
        "e",
        "h",
        "i",
        "o",
        "p",
        "r",
        "s",
        "t",
        "u",
        "x",
        "y",
        "«",
        "»",
        "А",
        "Б",
        "В",
        "Г",
        "Д",
        "Е",
        "Ж",
        "З",
        "И",
        "Й",
        "К",
        "Л",
        "М",
        "Н",
        "О",
        "П",
        "Р",
        "С",
        "Т",
        "У",
        "Ф",
        "Х",
        "Ц",
        "Ч",
        "Ш",
        "Щ",
        "Э",
        "Ю",
        "Я",
        "а",
        "б",
        "в",
        "г",
        "д",
        "е",
        "ж",
        "з",
        "и",
        "й",
        "к",
        "л",
        "м",
        "н",
        "о",
        "п",
        "р",
        "с",
        "т",
        "у",
        "ф",
        "х",
        "ц",
        "ч",
        "ш",
        "щ",
        "ъ",
        "ы",
        "ь",
        "э",
        "ю",
        "я",
        "ё",
        "№",
    ]

    def __init__(
        self,
        data_dir: Path,
        split: Literal["test", "train"],
        image_size: Tuple[int, int],
    ) -> None:
        self.image_size = image_size
        self.images: List[Tuple[Path, str]] = []
        images_dir = data_dir / split
        with open(data_dir / f"{split}.tsv") as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                image_path = images_dir / line[0]
                assert image_path.exists()
                self.images.append((image_path, line[1]))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path, target = self.images[idx]
        image = torchvision.io.read_image(
            str(image_path.resolve()), torchvision.io.ImageReadMode.RGB
        )
        image = random_pad(image, target_size=self.image_size, fill=0)
        image = image.to(torch.float) / 255.0

        target = torch.tensor([self.tokens.index(char) for char in target])
        return image, target

    def __len__(self) -> int:
        return len(self.images)


class CyrillicDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, image_size: Tuple[int, int]) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_dir = Path(__file__).parent / "data" / "Cyrillic"

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "constantinwerner/cyrillic-handwriting-dataset",
                path=self.data_dir,
                unzip=True,
                quiet=False,
            )

    def setup(self, stage: str = "fit") -> None:
        self.trainset = CyrillicDataset(
            self.data_dir, split="train", image_size=self.image_size
        )
        self.valset = CyrillicDataset(
            self.data_dir, split="test", image_size=self.image_size
        )

    def train_dataloader(self) -> DataLoader[Tuple[Tensor, bool]]:
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Tuple[Tensor, bool]]:
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn,
        )


HYPERPARAMS = {
    "max_steps": 90_000,
    "image_size": (64, 256),
    "batch_size": 64,
    "gradient_clip_val": 0.1,
    "backbone": {"name": "resnet50", "pretrained": True},
    "head_kwargs": {
        "num_channels": 256,
        "max_sequence_length": CyrillicDataset.max_length,
        "num_tokens": len(CyrillicDataset.tokens),
        "level": 5,
    },
    "optimizer": "AdamW",
    "optimizer_kwargs": {"lr": 1e-3, "weight_decay": 1e-2},
    "scheduler": "OneCycleLR",
    "scheduler_kwargs": {"max_lr": 1e-3, "total_steps": 90_000},
}

if __name__ == "__main__":
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=Path(__file__).parent / "logs", name="text_recognition"
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_steps=HYPERPARAMS["max_steps"],
        accelerator="gpu",
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        precision="16-mixed",
        gradient_clip_val=HYPERPARAMS["gradient_clip_val"],
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone(**HYPERPARAMS["backbone"])
        head = TextRecognition(
            in_channels=backbone.out_channels, **HYPERPARAMS["head_kwargs"]
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=None, heads=[head]),
            optimizer=getattr(torch.optim, HYPERPARAMS["optimizer"]),
            optimizer_kwargs=HYPERPARAMS["optimizer_kwargs"],
            scheduler=getattr(torch.optim.lr_scheduler, HYPERPARAMS["scheduler"]),
            scheduler_kwargs=HYPERPARAMS["scheduler_kwargs"],
            hyperparameters=HYPERPARAMS,
            data_config={"tokens": CyrillicDataset.tokens},
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
    trainer.fit(
        model,
        datamodule=CyrillicDataModule(
            batch_size=HYPERPARAMS["batch_size"], image_size=HYPERPARAMS["image_size"]
        ),
    )

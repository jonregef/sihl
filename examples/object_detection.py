from pathlib import Path
from typing import List, Dict, Tuple, Literal, Any
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

from sihl import SihlModel, SihlLightningModule, TorchvisionBackbone
from sihl.heads import ObjectDetection
from sihl.layers import FPN

coco_labels = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "trafficlight",
    "firehydrant",
    "streetsign",
    "stopsign",
    "parkingmeter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eyeglasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sportsball",
    "kite",
    "baseballbat",
    "baseballglove",
    "skateboard",
    "surfboard",
    "tennisracket",
    "bottle",
    "plate",
    "wineglass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hotdog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "mirror",
    "diningtable",
    "window",
    "desk",
    "toilet",
    "door",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cellphone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddybear",
    "hairdrier",
    "toothbrush",
    "hairbrush",
]

valid_coco_labels = coco_labels.copy()
for label in [
    "__background__",
    "streetsign",
    "hat",
    "shoe",
    "eyeglasses",
    "plate",
    "mirror",
    "window",
    "desk",
    "door",
    "blender",
    "hairbrush",
]:
    valid_coco_labels.remove(label)


def collate_fn(batch: List[Tuple[Tensor, Dict[Literal["labels", "boxes"], Any]]]):
    return (
        torch.stack([sample[0] for sample in batch]),
        {
            "classes": [
                (
                    torch.tensor(
                        [
                            valid_coco_labels.index(coco_labels[label])
                            for label in sample[1]["labels"]
                        ]
                    )
                    if "labels" in sample[1]
                    else torch.tensor([], dtype=torch.int64)
                )
                for sample in batch
            ],
            "boxes": [
                sample[1]["boxes"].data if "boxes" in sample[1] else torch.tensor([])
                for sample in batch
            ],
        },
    )


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(__file__).parent / "data"

    def prepare_data(self) -> None:
        if not (self.data_dir / "coco2017").exists():
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "awsaf49/coco-2017-dataset",
                path=self.data_dir,
                unzip=True,
                quiet=False,
            )

    def setup(self, stage: str = "") -> None:
        coco_data_dir = self.data_dir / "coco2017"
        self.trainset = torchvision.datasets.wrap_dataset_for_transforms_v2(
            torchvision.datasets.CocoDetection(
                str(coco_data_dir / "train2017"),
                str(coco_data_dir / "annotations" / "instances_train2017.json"),
                transforms=transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.Resize(800 - 1, max_size=800),
                        transforms.RandomCrop(800, pad_if_needed=True),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToDtype(torch.float32, scale=True),
                    ]
                ),
            )
        )
        self.valset = torchvision.datasets.wrap_dataset_for_transforms_v2(
            torchvision.datasets.CocoDetection(
                str(coco_data_dir / "val2017"),
                str(coco_data_dir / "annotations" / "instances_val2017.json"),
                transforms=transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.Resize(800 - 1, max_size=800),
                        transforms.RandomCrop(800, pad_if_needed=True),
                        transforms.ToDtype(torch.float32, scale=True),
                    ]
                ),
            )
        )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=Path(__file__).parent / "logs", name="object_detection"
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_steps=90_000,
        accelerator="gpu",
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        gradient_clip_val=3,
        precision="16-mixed",
        val_check_interval=0.25,
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone("resnet50", pretrained=True, frozen_levels=1)
        neck = FPN(backbone.out_channels, out_channels=256, bottom_level=3, top_level=7)
        head = ObjectDetection(
            neck.out_channels,
            num_classes=len(valid_coco_labels),
            soft_label_decay_steps=90_000,
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=neck, heads=[head]),
            optimizer=torch.optim.SGD,
            optimizer_kwargs={"lr": 1e-2, "weight_decay": 1e-4, "momentum": 0.9},
            scheduler=torch.optim.lr_scheduler.MultiStepLR,
            scheduler_kwargs={
                "milestones": [60_000, 80_000],
                "gamma": 0.1,
                "warmup_batches": 1000,
            },
            data_config={"categories": valid_coco_labels},
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
    trainer.fit(model, datamodule=CocoDataModule(batch_size=16))

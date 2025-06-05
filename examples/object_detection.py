from pathlib import Path
from typing import Tuple
import json
import logging

from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import tv_tensors
from torchvision.io import ImageReadMode
import kaggle
import lightning
import lightning.pytorch as pl
import torch
import torchvision
import torchvision.transforms.v2 as transforms

from sihl import SihlModel, SihlLightningModule, TorchvisionBackbone
from sihl.heads import ObjectDetection
import sihl.layers

lightning.seed_everything(0, workers=True)

ALL_COCO_LABELS = [
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

VALID_COCO_LABELS = ALL_COCO_LABELS.copy()
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
    VALID_COCO_LABELS.remove(label)


class CocoObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir: Path, train: bool = False, image_size: int = 640
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.data_dir = data_dir
        self.annots_by_image = {}
        split = "train" if train else "val"
        self.image_dir = self.data_dir / f"{split}2017"

        with open(
            self.data_dir / "annotations_trainval2017" / f"instances_{split}2017.json"
        ) as f:
            coco_data = json.load(f)

        image_by_id = {
            image_annot["id"]: image_annot["file_name"]
            for image_annot in coco_data["images"]
        }
        for annot in coco_data["annotations"]:
            if annot["iscrowd"]:
                continue
            image_name = image_by_id[annot["image_id"]]
            if not (self.image_dir / image_name).exists():
                continue
            image_path = str((self.image_dir / image_name).resolve())
            if image_path not in self.annots_by_image:
                self.annots_by_image[image_path] = {"boxes": [], "classes": []}
            x, y, w, h = annot["bbox"]
            self.annots_by_image[image_path]["boxes"].append([x, y, x + w, y + h])
            self.annots_by_image[image_path]["classes"].append(
                VALID_COCO_LABELS.index(ALL_COCO_LABELS[annot["category_id"]])
            )
        self.annots_by_image = list(self.annots_by_image.items())
        if train:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomPhotometricDistort(),
                    transforms.RandomZoomOut(side_range=(1.0, 2.0)),
                    transforms.Resize(image_size - 1, max_size=image_size),
                    transforms.RandomCrop(image_size, pad_if_needed=True),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.SanitizeBoundingBoxes(),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(image_size - 1, max_size=image_size),
                    transforms.RandomCrop(image_size, pad_if_needed=True),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path, annot = self.annots_by_image[idx]
        image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)
        annot = {
            "labels": torch.tensor(annot["classes"], dtype=torch.int64),
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor(annot["boxes"], dtype=torch.float32),
                format="xyxy",
                canvas_size=image.shape[1:],
            ),
        }
        image, annot = self.transforms(tv_tensors.Image(image), annot)
        return image, {"classes": annot["labels"], "boxes": annot["boxes"]}

    def __len__(self) -> int:
        return len(self.annots_by_image)

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([_[0] for _ in batch])
        classes = [_[1]["classes"] for _ in batch]
        boxes = [_[1]["boxes"] for _ in batch]
        return images, {"classes": classes, "boxes": boxes}


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(__file__).parent / "data" / "coco_2017"
        self.data_dir.mkdir(exist_ok=True)

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "clkmuhammed/microsoft-coco-2017-common-objects-in-context",
                path=self.data_dir,
                unzip=True,
                quiet=False,
            )

    def setup(self, stage: str = "") -> None:
        self.trainset = CocoObjectDetectionDataset(
            self.data_dir, train=True, image_size=HYPERPARAMS["image_size"]
        )
        self.validset = CocoObjectDetectionDataset(
            self.data_dir, train=False, image_size=HYPERPARAMS["image_size"]
        )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=CocoObjectDetectionDataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=CocoObjectDetectionDataset.collate_fn,
        )


HYPERPARAMS = {
    "max_steps": 90_000,
    "image_size": 640,
    "batch_size": 16,
    "gradient_clip_val": 0.1,
    "backbone": {"name": "resnet50", "pretrained": True, "frozen_levels": 1},
    "neck": "HybridEncoder",
    "neck_kwargs": {"out_channels": 256, "bottom_level": 3, "top_level": 7},
    "head_kwargs": {"num_channels": 256, "bottom_level": 3, "top_level": 7},
    "optimizer": "AdamW",
    "optimizer_kwargs": {"lr": 1e-4, "weight_decay": 1e-4, "backbone_lr_factor": 0.1},
    "scheduler": "MultiStepLR",
    "scheduler_kwargs": {"milestones": [60_000, 80_000], "gamma": 0.1},
}
if __name__ == "__main__":
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=Path(__file__).parent / "logs",
        default_hp_metric=False,
        name="object_detection",
    )

    trainer = pl.Trainer(
        max_steps=HYPERPARAMS["max_steps"],
        accelerator="gpu",
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        gradient_clip_val=HYPERPARAMS["gradient_clip_val"],
        precision="16-mixed",
        val_check_interval=0.25,
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone(**HYPERPARAMS["backbone"])
        neck = getattr(sihl.layers, HYPERPARAMS["neck"])(
            backbone.out_channels, **HYPERPARAMS["neck_kwargs"]
        )
        head = ObjectDetection(
            in_channels=neck.out_channels,
            num_classes=len(VALID_COCO_LABELS),
            **HYPERPARAMS["head_kwargs"],
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=neck, heads=[head]),
            optimizer=getattr(torch.optim, HYPERPARAMS["optimizer"]),
            optimizer_kwargs=HYPERPARAMS["optimizer_kwargs"],
            scheduler=getattr(torch.optim.lr_scheduler, HYPERPARAMS["scheduler"]),
            scheduler_kwargs=HYPERPARAMS["scheduler_kwargs"],
            hyperparameters=HYPERPARAMS,
            data_config={"categories": VALID_COCO_LABELS},
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
    trainer.fit(model, datamodule=CocoDataModule(batch_size=HYPERPARAMS["batch_size"]))

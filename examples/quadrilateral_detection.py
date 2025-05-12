from pathlib import Path
from typing import Tuple, List, Literal, Dict, Any
import logging
import random
import sys
import xml.etree.ElementTree as ET

from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import tv_tensors
import kaggle
import lightning
import lightning.pytorch as pl
import torch
import torchvision
import torchvision.transforms.v2 as transforms

sys.path.append("../src")

from sihl import SihlModel, SihlLightningModule, TorchvisionBackbone
from sihl.heads import QuadrilateralDetection


AIRCRAFT_LABELS = [f"A{n}" for n in range(1, 21)]


### HACK: torchvision doesn't support augmenting keypoints yet
# https://pytorch.org/blog/extending-torchvisions-transforms-to-object-detection-segmentation-and-video-tasks/#development-milestones-and-future-work
# So, we'll convert each keypoint to a tiny square bbox
# we'll augment those then convert them back by taking their center point
def polygons_to_bboxes(polygons: Tensor, canvas_size: Tuple[int, int]):
    """polygons: (N, K, 2)"""
    box_centers = polygons.reshape((-1, 2))
    flat_boxes = torch.cat([box_centers, torch.ones_like(box_centers)], dim=1)
    return tv_tensors.BoundingBoxes(
        flat_boxes, format=tv_tensors.BoundingBoxFormat.CXCYWH, canvas_size=canvas_size
    )


def bboxes_to_polygons(bboxes: tv_tensors.BoundingBoxes, num_vertices: int) -> Tensor:
    return bboxes[:, :2].reshape((-1, num_vertices, 2))


###


def collate_fn(batch: List[Tuple[Tensor, Dict[Literal["classes", "boxes"], Any]]]):
    return (
        torch.stack([sample[0] for sample in batch]),
        {
            "classes": [
                (torch.tensor([AIRCRAFT_LABELS.index(_) for _ in sample[1]["classes"]]))
                for sample in batch
            ],
            "quads": [sample[1]["quads"] for sample in batch],
        },
    )


def parse_xml_annot(raw: str) -> Tuple[List[str], Tensor]:
    root = ET.fromstring(raw)
    categories = []
    quads = []
    for obj in root.findall("object"):
        categories.append(obj.find("name").text)
        quad = obj.find("robndbox")
        x1 = float(quad.find("x_left_top").text)
        y1 = float(quad.find("y_left_top").text)
        x2 = float(quad.find("x_right_top").text)
        y2 = float(quad.find("y_right_top").text)
        x3 = float(quad.find("x_right_bottom").text)
        y3 = float(quad.find("y_right_bottom").text)
        x4 = float(quad.find("x_left_bottom").text)
        y4 = float(quad.find("y_left_bottom").text)
        quads.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return categories, torch.tensor(quads)


class AircraftDataset(torch.utils.data.Dataset[Tuple[Tensor, bool]]):
    def __init__(self, data_dir: Path, train: bool, indices: List[str]) -> None:
        self.train = train
        self.images: List[Tuple[Path, Tuple[str, Tensor]]] = []
        images_dir = data_dir / "JPEGImages"
        annots_dir = data_dir / "Annotations" / "Oriented Bounding Boxes"
        # split = "train" if self.train else "test"
        # with open(data_dir / "ImageSets" / "Main" / f"{split}.txt") as f:
        #     idxs = f.read().splitlines()
        for idx in indices:
            with open(annots_dir / f"{idx}.xml") as f:
                categories, quads = parse_xml_annot(f.read())
            self.images.append(
                (images_dir / f"{idx}.jpg", {"classes": categories, "quads": quads})
            )
        self.transforms = None
        # if self.train:
        # https://github.com/pytorch/vision/issues/566#issuecomment-814221986
        if self.train:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(512 - 1, max_size=512),
                    transforms.RandomCrop(512, pad_if_needed=True),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomChoice(
                        [
                            transforms.RandomRotation((90, 90)),
                            transforms.RandomRotation((180, 180)),
                            transforms.RandomRotation((270, 270)),
                            transforms.Identity(),
                        ]
                    ),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(512 - 1, max_size=512),
                    transforms.RandomCrop(512, pad_if_needed=True),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path, target = self.images[idx]
        image = torchvision.io.read_image(
            str(image_path.resolve()), torchvision.io.ImageReadMode.RGB
        )
        quad_boxes = polygons_to_bboxes(target["quads"], image.shape[1:])
        image, quad_boxes = self.transforms(image, quad_boxes)
        target["quads"] = bboxes_to_polygons(quad_boxes, num_vertices=4)
        return image, target

    def __len__(self) -> int:
        return len(self.images)


class AircraftDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(__file__).parent / "data" / "military_aircraft_recognition"

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "khlaifiabilel/military-aircraft-recognition-dataset",
                path=self.data_dir,
                unzip=True,
                quiet=False,
            )

    def setup(self, stage: str = "") -> None:
        # Re-split to 90/10
        all_idxs = [_.stem for _ in (self.data_dir / "JPEGImages").glob("*.jpg")]
        random.shuffle(all_idxs)
        split_idxs = int(0.1 * len(all_idxs))
        train_idxs, valid_idxs = all_idxs[split_idxs:], all_idxs[:split_idxs]
        self.trainset = AircraftDataset(self.data_dir, train=True, indices=train_idxs)
        self.valset = AircraftDataset(self.data_dir, train=False, indices=valid_idxs)

    def train_dataloader(self) -> DataLoader[Tuple[Tensor, bool]]:
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Tuple[Tensor, bool]]:
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )


HYPERPARAMS = {
    "max_steps": 10_000,
    "image_size": 640,
    "batch_size": 16,
    "gradient_clip_val": 1,
    "backbone": {"name": "resnet50", "pretrained": True, "frozen_levels": 1},
    "neck": None,
    "neck_kwargs": {},
    "head_kwargs": {
        "num_classes": len(AIRCRAFT_LABELS),
        "num_channels": 256,
        "bottom_level": 5,
        "top_level": 5,
    },
    "optimizer": "AdamW",
    "optimizer_kwargs": {"lr": 1e-4, "weight_decay": 1e-4},
    "scheduler": "MultiStepLR",
    "scheduler_kwargs": {"milestones": [7500], "gamma": 0.1},
}
if __name__ == "__main__":
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=Path(__file__).parent / "logs", name="quadrilateral_detection"
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_steps=HYPERPARAMS["max_steps"],
        accelerator="gpu",
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        gradient_clip_val=HYPERPARAMS["gradient_clip_val"],
        precision="16-mixed",
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone(**HYPERPARAMS["backbone"])
        head = QuadrilateralDetection(
            backbone.out_channels, **HYPERPARAMS["head_kwargs"]
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=None, heads=[head]),
            optimizer=getattr(torch.optim, HYPERPARAMS["optimizer"]),
            optimizer_kwargs=HYPERPARAMS["optimizer_kwargs"],
            scheduler=getattr(torch.optim.lr_scheduler, HYPERPARAMS["scheduler"]),
            scheduler_kwargs=HYPERPARAMS["scheduler_kwargs"],
            hyperparameters=HYPERPARAMS,
            data_config={"categories": AIRCRAFT_LABELS},
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
    trainer.fit(
        model, datamodule=AircraftDataModule(batch_size=HYPERPARAMS["batch_size"])
    )

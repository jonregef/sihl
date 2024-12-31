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
from sihl.heads import KeypointDetection
from sihl.layers import FPN


### HACK: torchvision doesn't support augmenting keypoints yet
# https://pytorch.org/blog/extending-torchvisions-transforms-to-object-detection-segmentation-and-video-tasks/#development-milestones-and-future-work
# So, we'll convert each keypoint to a tiny square bbox
# we'll augment those then convert them back by taking their center point
def polygons_to_bboxes(polygons: Tensor, canvas_size: Tuple[int, int]):
    """polygons: (N, K, 2)"""
    SIZE = 3  # bounding box size in pixels
    box_centers = polygons.reshape((-1, 2))
    flat_boxes = torch.cat([box_centers, torch.full_like(box_centers, SIZE)], dim=1)
    return tv_tensors.BoundingBoxes(
        flat_boxes, format=tv_tensors.BoundingBoxFormat.CXCYWH, canvas_size=canvas_size
    )


def bboxes_to_polygons(bboxes: tv_tensors.BoundingBoxes, num_vertices: int) -> Tensor:
    widths, heights = bboxes[:, 2], bboxes[:, 3]
    inside = ~((widths == 0) | (heights == 0)).reshape((-1, num_vertices))
    return bboxes[:, :2].reshape((-1, num_vertices, 2)), inside


###


class CocoHumanPoseEstimationDataset(torch.utils.data.Dataset):
    KEYPOINT_LABELS = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    SKELETON = [
        ["left_ankle", "left_knee"],
        ["left_knee", "left_hip"],
        ["right_ankle", "right_knee"],
        ["right_knee", "right_hip"],
        ["left_hip", "right_hip"],
        ["left_shoulder", "left_hip"],
        ["right_shoulder", "right_hip"],
        ["left_shoulder", "right_shoulder"],
        ["left_shoulder", "left_elbow"],
        ["right_shoulder", "right_elbow"],
        ["left_elbow", "left_wrist"],
        ["right_elbow", "right_wrist"],
        ["left_eye", "right_eye"],
        ["nose", "left_eye"],
        ["nose", "right_eye"],
        ["left_eye", "left_ear"],
        ["right_eye", "right_ear"],
        ["left_ear", "left_shoulder"],
        ["right_ear", "right_shoulder"],
    ]

    def __init__(self, data_dir: Path, train: bool = False) -> None:
        self.image_size = 800
        self.train = train
        self.data_dir = data_dir
        self.annots_by_image = {}
        split = "train" if train else "val"
        self.image_dir = self.data_dir / f"{split}2017"

        with open(
            self.data_dir
            / "annotations_trainval2017"
            / f"person_keypoints_{split}2017.json"
        ) as f:
            coco_data = json.load(f)

        image_by_id = {
            image_annot["id"]: image_annot["file_name"]
            for image_annot in coco_data["images"]
        }
        for annot in coco_data["annotations"]:
            image_name = image_by_id[annot["image_id"]]
            if not (self.image_dir / image_name).exists():
                continue
            image_path = str((self.image_dir / image_name).resolve())
            if image_path not in self.annots_by_image:
                self.annots_by_image[image_path] = []
            self.annots_by_image[image_path].append(annot["keypoints"])
        self.annots_by_image = list(self.annots_by_image.items())
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size - 1, max_size=self.image_size),
                transforms.RandomCrop(self.image_size, pad_if_needed=True),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path, annot = self.annots_by_image[idx]
        image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)
        target = torch.tensor(annot).reshape((-1, len(self.KEYPOINT_LABELS), 3))
        keypoints = target[:, :, :2].to(torch.float32)
        presence = target[:, :, 2].to(torch.bool)
        keypoints = polygons_to_bboxes(keypoints, image.shape[1:])
        image, keypoints = self.transform(image, keypoints)
        keypoints, inside = bboxes_to_polygons(keypoints, len(self.KEYPOINT_LABELS))
        presence = presence & inside
        keypoints = keypoints * presence.unsqueeze(2)
        # remove instances that have no present keypoint
        visible_instances = presence.any(dim=1)
        presence = presence[visible_instances]
        keypoints = keypoints[visible_instances]
        return image, {"keypoints": keypoints, "presence": presence}

    def __len__(self) -> int:
        return len(self.annots_by_image)

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([_[0] for _ in batch])
        keypoints = [_[1]["keypoints"] for _ in batch]
        presence = [_[1]["presence"] for _ in batch]
        return images, {"keypoints": keypoints, "presence": presence}


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32) -> None:
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
        self.trainset = CocoHumanPoseEstimationDataset(self.data_dir, train=True)
        self.validset = CocoHumanPoseEstimationDataset(self.data_dir, train=False)

    def train_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=CocoHumanPoseEstimationDataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
            collate_fn=CocoHumanPoseEstimationDataset.collate_fn,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=Path(__file__).parent / "logs",
        name="keypoint_detection",
        # version="tmp",
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_steps=90_000,
        accelerator="gpu",
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        gradient_clip_val=1,
        precision="16-mixed",
        val_check_interval=1000,
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone("resnet50", pretrained=True, frozen_levels=1)
        neck = FPN(backbone.out_channels, 256, bottom_level=3, top_level=7)
        head = KeypointDetection(
            in_channels=neck.out_channels,
            num_keypoints=len(CocoHumanPoseEstimationDataset.KEYPOINT_LABELS),
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
            data_config={
                "keypoints": CocoHumanPoseEstimationDataset.KEYPOINT_LABELS,
                "links": CocoHumanPoseEstimationDataset.SKELETON,
            },
        )

    log.debug(
        summary(
            model,
            row_settings=("var_names",),
            col_names=("num_params", "trainable"),
            verbose=0,
            depth=5,
        )
    )
    trainer.fit(model, datamodule=CocoDataModule(batch_size=16))

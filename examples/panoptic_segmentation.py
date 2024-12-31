from typing import Tuple
from pathlib import Path
import logging
import sys
import json

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

sys.path.append("../src")

from sihl import SihlModel, SihlLightningModule, TorchvisionBackbone
from sihl.heads import PanopticSegmentation
from sihl.layers import FPN


class CocoPanopticSegmentationDataset(torch.utils.data.Dataset):
    # FIXME: hardcoded "ignore" category for input padding?
    THING_CATEGORIES = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
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
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
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
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    STUFF_CATEGORIES = [
        "padding",
        "banner",
        "blanket",
        "bridge",
        "cardboard",
        "counter",
        "curtain",
        "door-stuff",
        "floor-wood",
        "flower",
        "fruit",
        "gravel",
        "house",
        "light",
        "mirror-stuff",
        "net",
        "pillow",
        "platform",
        "playingfield",
        "railroad",
        "river",
        "road",
        "roof",
        "sand",
        "sea",
        "shelf",
        "snow",
        "stairs",
        "tent",
        "towel",
        "wall-brick",
        "wall-stone",
        "wall-tile",
        "wall-wood",
        "water-other",
        "window-blind",
        "window-other",
        "tree-merged",
        "fence-merged",
        "ceiling-merged",
        "sky-other-merged",
        "cabinet-merged",
        "table-merged",
        "floor-other-merged",
        "pavement-merged",
        "mountain-merged",
        "grass-merged",
        "dirt-merged",
        "paper-merged",
        "food-other-merged",
        "building-other-merged",
        "rock-merged",
        "wall-other-merged",
        "rug-merged",
    ]
    LABEL_MAP = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
        92: "banner",
        93: "blanket",
        95: "bridge",
        100: "cardboard",
        107: "counter",
        109: "curtain",
        112: "door-stuff",
        118: "floor-wood",
        119: "flower",
        122: "fruit",
        125: "gravel",
        128: "house",
        130: "light",
        133: "mirror-stuff",
        138: "net",
        141: "pillow",
        144: "platform",
        145: "playingfield",
        147: "railroad",
        148: "river",
        149: "road",
        151: "roof",
        154: "sand",
        155: "sea",
        156: "shelf",
        159: "snow",
        161: "stairs",
        166: "tent",
        168: "towel",
        171: "wall-brick",
        175: "wall-stone",
        176: "wall-tile",
        177: "wall-wood",
        178: "water-other",
        180: "window-blind",
        181: "window-other",
        184: "tree-merged",
        185: "fence-merged",
        186: "ceiling-merged",
        187: "sky-other-merged",
        188: "cabinet-merged",
        189: "table-merged",
        190: "floor-other-merged",
        191: "pavement-merged",
        192: "mountain-merged",
        193: "grass-merged",
        194: "dirt-merged",
        195: "paper-merged",
        196: "food-other-merged",
        197: "building-other-merged",
        198: "rock-merged",
        199: "wall-other-merged",
        200: "rug-merged",
    }
    ALL_CATEGORIES = STUFF_CATEGORIES + THING_CATEGORIES
    PAD = STUFF_CATEGORIES.index("padding")
    IGNORE = len(ALL_CATEGORIES)

    def __init__(self, data_dir: Path, train: bool = False) -> None:
        self.image_size = 640  # 800
        self.train = train
        self.data_dir = data_dir
        self.annots_by_image = {}
        split = "train" if train else "val"
        self.image_dir = self.data_dir / f"{split}2017"
        self.annot_dir = (
            self.data_dir
            / "panoptic_annotations_trainval2017"
            / f"panoptic_{split}2017"
        )

        with open(
            self.data_dir
            / "panoptic_annotations_trainval2017"
            / f"panoptic_{split}2017.json"
        ) as f:
            coco_data = json.load(f)

        image_by_id = {
            image_annot["id"]: image_annot["file_name"]
            for image_annot in coco_data["images"]
        }
        for annot in coco_data["annotations"]:
            image_name = image_by_id[annot["image_id"]]
            if (
                not (self.image_dir / image_name).exists()
                or not (self.annot_dir / annot["file_name"]).exists()
            ):
                continue
            image_path = str((self.image_dir / image_name).resolve())
            self.annots_by_image[image_path] = annot
        self.annots_by_image = [
            (key, value) for key, value in self.annots_by_image.items()
        ]

        if self.train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size - 1, max_size=self.image_size),
                    transforms.RandomCrop(
                        self.image_size,
                        pad_if_needed=True,
                        fill={tv_tensors.Image: 0, tv_tensors.Mask: self.PAD},
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size - 1, max_size=self.image_size),
                    transforms.RandomCrop(
                        self.image_size,
                        pad_if_needed=True,
                        fill={tv_tensors.Image: 0, tv_tensors.Mask: self.PAD},
                    ),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path, annot = self.annots_by_image[idx]
        image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)
        target = torchvision.io.read_image(
            str((self.annot_dir / annot["file_name"]).resolve()), mode=ImageReadMode.RGB
        )
        r, g, b = target.to(torch.int64)
        id_map = r + 256 * g + 256**2 * b
        semantic_map = id_map.clone()
        semantic_map[semantic_map == 0] = self.IGNORE
        for segment in annot["segments_info"]:
            category = self.ALL_CATEGORIES.index(self.LABEL_MAP[segment["category_id"]])
            semantic_map[semantic_map == segment["id"]] = category
        target = torch.stack([semantic_map, id_map])
        image, target = self.transform(image, tv_tensors.Mask(target))
        return image, target.to(torch.int64)

    def __len__(self) -> int:
        return len(self.annots_by_image)


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
        self.trainset = CocoPanopticSegmentationDataset(self.data_dir, train=True)
        self.validset = CocoPanopticSegmentationDataset(self.data_dir, train=False)

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
        save_dir=Path(__file__).parent / "logs", name="panoptic_segmentation"
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_steps=90_000,
        accelerator="gpu",
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        gradient_clip_val=1,
        precision="16-mixed",
        val_check_interval=0.25,
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone("resnet50", pretrained=True, frozen_levels=1)
        neck = FPN(backbone.out_channels, out_channels=256, bottom_level=3, top_level=7)
        head = PanopticSegmentation(
            in_channels=neck.out_channels,
            num_stuff_classes=len(CocoPanopticSegmentationDataset.STUFF_CATEGORIES),
            num_thing_classes=len(CocoPanopticSegmentationDataset.THING_CATEGORIES),
            mask_top_level=5,
            soft_label_decay_steps=90_000,
            ignore_index=CocoPanopticSegmentationDataset.IGNORE,
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=neck, heads=[head]),
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-4},
            data_config={
                "categories": CocoPanopticSegmentationDataset.ALL_CATEGORIES,
                "ignore": CocoPanopticSegmentationDataset.IGNORE,
            },
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

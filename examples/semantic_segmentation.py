from pathlib import Path
from typing import Tuple
import json
import logging

from faster_coco_eval.core.mask import decode as mask_decode
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
from sihl.layers import FPN
from sihl.heads import SemanticSegmentation


class CocoSemanticSegmentationDataset(torch.utils.data.Dataset):
    CATEGORIES = [
        "background",
        "banner",
        "blanket",
        "branch",
        "bridge",
        "building-other",
        "bush",
        "cabinet",
        "cage",
        "cardboard",
        "carpet",
        "ceiling-other",
        "ceiling-tile",
        "cloth",
        "clothes",
        "clouds",
        "counter",
        "cupboard",
        "curtain",
        "desk-stuff",
        "dirt",
        "door-stuff",
        "fence",
        "floor-marble",
        "floor-other",
        "floor-stone",
        "floor-tile",
        "floor-wood",
        "flower",
        "fog",
        "food-other",
        "fruit",
        "furniture-other",
        "grass",
        "gravel",
        "ground-other",
        "hill",
        "house",
        "leaves",
        "light",
        "mat",
        "metal",
        "mirror-stuff",
        "moss",
        "mountain",
        "mud",
        "napkin",
        "net",
        "paper",
        "pavement",
        "pillow",
        "plant-other",
        "plastic",
        "platform",
        "playingfield",
        "railing",
        "railroad",
        "river",
        "road",
        "rock",
        "roof",
        "rug",
        "salad",
        "sand",
        "sea",
        "shelf",
        "sky-other",
        "skyscraper",
        "snow",
        "solid-other",
        "stairs",
        "stone",
        "straw",
        "structural-other",
        "table",
        "tent",
        "textile-other",
        "towel",
        "tree",
        "vegetable",
        "wall-brick",
        "wall-concrete",
        "wall-other",
        "wall-panel",
        "wall-stone",
        "wall-tile",
        "wall-wood",
        "water-other",
        "waterdrops",
        "window-blind",
        "window-other",
        "wood",
        "other",
    ]

    def __init__(self, data_dir: Path, train: bool = False) -> None:
        self.image_size = 640
        self.train = train
        self.data_dir = data_dir
        self.annots_by_image = {}
        split = "train" if train else "val"
        self.image_dir = self.data_dir / f"{split}2017"

        with open(
            self.data_dir / "stuff_annotations_trainval2017" / f"stuff_{split}2017.json"
        ) as f:
            coco_data = json.load(f)

        image_by_id = {
            image_annot["id"]: image_annot["file_name"]
            for image_annot in coco_data["images"]
        }
        for annot in coco_data["annotations"]:
            image_name = image_by_id[annot["image_id"]]
            image_path = str((self.image_dir / image_name).resolve())
            if image_path not in self.annots_by_image:
                self.annots_by_image[image_path] = []
            self.annots_by_image[image_path].append(annot)
        self.annots_by_image = [
            (key, value) for key, value in self.annots_by_image.items()
        ]

        if self.train:
            self.transform = transforms.Compose(
                [
                    transforms.ScaleJitter(
                        (self.image_size, self.image_size), scale_range=(0.5, 2.0)
                    ),
                    transforms.RandomCrop(self.image_size, pad_if_needed=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size - 1, max_size=self.image_size),
                    transforms.RandomCrop(self.image_size, pad_if_needed=True),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path, annots = self.annots_by_image[idx]
        image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)
        target = torch.stack(
            [
                torch.from_numpy(mask_decode(annot["segmentation"]))
                * (annot["category_id"] - 91)
                for annot in annots
            ]
        ).sum(dim=0)
        image, target = self.transform(image, tv_tensors.Mask(target))
        return image, target

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
        self.trainset = CocoSemanticSegmentationDataset(self.data_dir, train=True)
        self.validset = CocoSemanticSegmentationDataset(self.data_dir, train=False)

    def train_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> DataLoader[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self.validset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


if __name__ == "__main__":
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=Path(__file__).parent / "logs", name="semantic_segmentation"
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_steps=90_000,
        accelerator="gpu",
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        gradient_clip_val=1,
        val_check_interval=0.2,
        precision="16-mixed",
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone("resnet50", pretrained=True)
        neck = FPN(backbone.out_channels, 256, bottom_level=3, top_level=5)
        head = SemanticSegmentation(
            in_channels=neck.out_channels,
            num_classes=len(CocoSemanticSegmentationDataset.CATEGORIES),
            ignore_index=CocoSemanticSegmentationDataset.CATEGORIES.index("other"),
            bottom_level=3,
            top_level=5,
            num_layers=0,
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=neck, heads=[head]),
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-4},
            data_config={
                "categories": CocoSemanticSegmentationDataset.CATEGORIES,
                "ignore": CocoSemanticSegmentationDataset.CATEGORIES.index("other"),
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

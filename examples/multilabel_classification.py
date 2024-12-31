from pathlib import Path
import logging
import sys

from rich.logging import RichHandler
from torch import Tensor
from torch.nn import functional
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
from sihl.heads import MultilabelClassification

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
for _ in [
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
    valid_coco_labels.remove(_)


def collate_fn(batch):
    images = torch.stack([_[0] for _ in batch])
    labels = torch.stack(
        [
            functional.one_hot(
                torch.tensor(
                    [
                        valid_coco_labels.index(coco_labels[label])
                        for label in sample[1].get("labels", [])
                    ],
                    dtype=torch.int64,
                ).unique(),
                num_classes=len(valid_coco_labels),
            ).sum(dim=0)
            for sample in batch
        ]
    )
    return images, labels


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(__file__).parent / "data"

    def prepare_data(self) -> None:
        if not (self.data_dir / "coco2017").exists():
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "awsaf49/coco-2017-dataset", path=self.data_dir, unzip=True, quiet=False
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
            ),
            target_keys=["labels"],
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
            ),
            target_keys=["labels"],
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
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    logger = pl.loggers.TensorBoardLogger(
        save_dir=Path(__file__).parent / "logs", name="multilabel_classification"
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        gradient_clip_val=1,
        precision="16-mixed",
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone("resnet50", pretrained=True)
        head = MultilabelClassification(
            in_channels=backbone.out_channels, num_labels=len(valid_coco_labels)
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=None, heads=[head]),
            optimizer=torch.optim.SGD,
            optimizer_kwargs={"lr": 1e-2, "weight_decay": 1e-4, "momentum": 0.9},
            data_config={"categories": valid_coco_labels},
        )

    log.info(
        summary(
            model.model,
            input_data=model.model.backbone.dummy_input,
            row_settings=("var_names",),
            col_names=("output_size", "num_params", "mult_adds", "trainable"),
            verbose=0,
            depth=4,
        )
    )
    trainer.fit(model, datamodule=CocoDataModule(batch_size=16))

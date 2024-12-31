from pathlib import Path
from typing import Tuple, List

from PIL import Image
from rich.logging import RichHandler
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import tv_tensors
import kaggle
import lightning
import lightning.pytorch as pl
import logging
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms

from sihl import SihlModel, SihlLightningModule
from sihl.heads import DepthEstimation
from sihl.layers import FPN
from sihl.torchvision_backbone import TorchvisionBackbone


class NYU2Dataset(torch.utils.data.Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, data_dir: Path, train: bool = False) -> None:
        self.train = train
        self.data_dir = data_dir
        self.min_dist, self.max_dist = 0.0, 10.0
        self.images: List[Tuple[Path, List[float]]] = []
        for imgf in data_dir.rglob("rgb_*.png"):
            depthf = imgf.with_name(imgf.name.replace("rgb", "depth"))
            if not depthf.exists():
                continue
            self.images.append((imgf, depthf))
        if self.train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((512, 640), pad_if_needed=True),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop((512, 640), pad_if_needed=True),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path, target_path = self.images[idx]
        image = torchvision.io.read_image(str(image_path.resolve()))
        # images are in uint16 format, but we cast to int32 for pytorch compatibility
        target = torch.tensor(np.array(Image.open(target_path)).astype(np.int32))
        image, target = self.transform(tv_tensors.Image(image), tv_tensors.Mask(target))
        mask = target != 0
        target = (target - 1) / (2**16 - 2)
        target = target * (self.max_dist - self.min_dist) + self.min_dist
        return image, {"targets": target, "masks": mask}

    def __len__(self) -> int:
        return len(self.images)


class NYU2DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(__file__).parent / "data" / "NYU2"

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "awsaf49/nyuv2-official-split-dataset",
                path=self.data_dir,
                unzip=True,
                quiet=False,
            )

    def setup(self, stage: str = "") -> None:
        self.trainset = NYU2Dataset(self.data_dir / "train", train=True)
        self.valset = NYU2Dataset(self.data_dir / "test", train=False)

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
        save_dir=Path(__file__).parent / "logs", name="depth_estimation"
    )
    lightning.seed_everything(0)

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=[0],
        logger=logger,
        callbacks=[pl.callbacks.RichProgressBar(leave=True)],
        precision="16-mixed",
        gradient_clip_val=1,
    )
    with trainer.init_module():
        backbone = TorchvisionBackbone("resnet50", pretrained=True, top_level=7)
        neck = FPN(backbone.out_channels, 256, bottom_level=2, top_level=7)
        head = DepthEstimation(
            in_channels=neck.out_channels,
            lower_bound=0.0,
            upper_bound=10.0,
            bottom_level=2,
            top_level=7,
        )
        model = SihlLightningModule(
            SihlModel(backbone=backbone, neck=neck, heads=[head]),
            optimizer=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-4},
            data_config={"range": (0.0, 10.0), "ignore": 0},
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
    trainer.fit(model, datamodule=NYU2DataModule(batch_size=32))

from copy import deepcopy
from typing import Any
from typing import Tuple, List, Dict, Union, Type, Optional
import warnings

from lightning.fabric.utilities.exceptions import MisconfigurationException
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import lightning.pytorch as pl
import torch

from sihl.heads import Head, ViewInvarianceLearning
from sihl.sihl_model import SihlModel
from sihl.visualization import visualize

# https://github.com/Lightning-AI/pytorch-lightning/issues/5558#issuecomment-1199306489
warnings.filterwarnings("ignore", "Detected call of", UserWarning)
torch.set_float32_matmul_precision("high")


class SihlLightningModule(pl.LightningModule):
    log_kwargs = {"on_epoch": False, "on_step": True, "prog_bar": True}

    def __init__(
        self,
        model: SihlModel,
        optimizer: Type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        data_config: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.data_config = data_config or [{} for _ in range(len(self.model.heads))]
        if isinstance(self.data_config, dict):
            self.data_config = [self.data_config]
        self.visualization_batches = []
        self.visualize_n_batches = 5
        self.visualize_n_per_batch = 1

    def forward(self, x: Tensor) -> List[Union[Tensor, List[Tensor]]]:
        return self.model(x)

    def on_fit_start(self):
        self = self.to(memory_format=torch.channels_last)

    def on_train_epoch_start(self) -> None:
        for head in self.model.heads:
            head: Head
            if hasattr(head, "on_train_epoch_start"):
                head.on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        for head in self.model.heads:
            head: Head
            if hasattr(head, "on_train_epoch_end"):
                head.on_train_epoch_end()

    def training_step(
        self, batch: Union[Tuple[Tensor, Any], Tensor], batch_idx: int
    ) -> Tensor:
        x, targets = (batch, None) if isinstance(batch, Tensor) else batch
        # FIXME: heads' registered buffers aren't moved to device by default
        self.to(x.device)
        if not isinstance(targets, list):
            targets = [targets]  # single-headed

        head_inputs = self.model.extract_features(x)
        losses = []
        for head_idx, (head, target) in enumerate(zip(self.model.heads, targets)):
            head: Head
            if isinstance(head, ViewInvarianceLearning):  # FIXME: this is a hack
                target = self.model.extract_features(target)  # second view of inputs

            if isinstance(target, dict):
                loss, metrics = head.training_step(head_inputs, **target)
            else:
                loss, metrics = head.training_step(head_inputs, target)

            metrics = {f"{head_idx}/train/{k}": v for k, v in metrics.items()}
            try:
                self.log(f"{head_idx}/train/loss", loss, **self.log_kwargs)
                self.log_dict(metrics, **self.log_kwargs)
            except MisconfigurationException:
                pass
            losses.append(loss)
        loss = torch.stack(losses).sum()
        scheduler = self.lr_schedulers()
        try:
            if isinstance(scheduler, LRScheduler):
                self.log("learning_rate", scheduler.get_last_lr()[0], **self.log_kwargs)
            else:
                lightning_optimizer = self.optimizers()
                for param_group in lightning_optimizer.optimizer.param_groups:
                    self.log("learning_rate", param_group["lr"], **self.log_kwargs)
        except (MisconfigurationException, AttributeError):
            pass
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch: Tuple[Tensor, Any], batch_idx: int) -> Tensor:
        x, targets = (batch, None) if isinstance(batch, Tensor) else batch
        if not isinstance(targets, list):
            targets = [targets]

        if len(self.visualization_batches) < self.visualize_n_batches:
            batch_idxs = [_["idx"] for _ in self.visualization_batches]
            if batch_idx not in batch_idxs:
                self.visualization_batches.append(
                    {
                        "input": x[: self.visualize_n_per_batch],
                        "targets": targets,
                        "idx": batch_idx,
                    }
                )

        head_inputs = self.model.extract_features(x)
        total_loss = 0
        for head_idx, (head, target) in enumerate(zip(self.model.heads, targets)):
            head: Head
            if isinstance(head, ViewInvarianceLearning):  # FIXME: this is a hack
                target = self.model.extract_features(target)

            if isinstance(target, dict):
                head_loss, metrics = head.validation_step(head_inputs, **target)
            else:
                head_loss, metrics = head.validation_step(head_inputs, target)

            total_loss = head_loss + total_loss
            metrics = {f"{head_idx}/valid/{k}": v for k, v in metrics.items()}
            try:
                self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
            except MisconfigurationException:
                pass
        torch.cuda.empty_cache()
        return total_loss

    def pretraining_step(self, batch: Tuple[Tensor, Any]) -> None:
        x, targets = (batch, None) if isinstance(batch, Tensor) else batch
        if not isinstance(targets, list):
            targets = [targets]

        head_inputs = self.model.extract_features(x.to("cuda"))  # FIXME
        for head_idx, (head, target) in enumerate(zip(self.model.heads, targets)):
            head: Head
            if not hasattr(head, "pretraining_step"):
                continue
            if isinstance(head, ViewInvarianceLearning):  # FIXME: this is a hack
                target = self.model.extract_features(target)

            if isinstance(target, dict):
                head.pretraining_step(head_inputs, **target)
            else:
                head.pretraining_step(head_inputs, target)

        torch.cuda.empty_cache()

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[List[Optimizer], List[LRScheduler]]]:
        optimizer_kwargs = deepcopy(self.optimizer_kwargs)

        # It can be beneficial to use a lower lr for the backbone when finetuning
        base_lr, backbone_lr_factor = optimizer_kwargs.get("lr", 1e-3), 1.0
        if "backbone_lr_factor" in optimizer_kwargs:
            backbone_lr_factor = optimizer_kwargs.pop("backbone_lr_factor")

        # backbone_module_names = {mn for mn, m in self.model.backbone.named_modules()}
        backbone_params = set(self.model.backbone.parameters())
        non_backbone_params = set(self.parameters()) - backbone_params
        parameters = [
            {"params": list(non_backbone_params)},
            {"params": list(backbone_params), "lr": base_lr * backbone_lr_factor},
        ]

        if "weight_decay" in optimizer_kwargs:
            # Certain layers should not be subject to weight decay, so we isolate them
            blacklist = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.Embedding)
            no_decay = set()
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    if pn.endswith("bias") or isinstance(m, blacklist):
                        no_decay.add(p)
            decay = set(self.parameters()) - no_decay
            parameters = [
                {"params": list(non_backbone_params & decay)},
                {"params": list(non_backbone_params & no_decay), "weight_decay": 0.0},
                {
                    "params": list(backbone_params & decay),
                    "lr": base_lr * backbone_lr_factor,
                },
                {
                    "params": list(backbone_params & no_decay),
                    "lr": base_lr * backbone_lr_factor,
                    "weight_decay": 0.0,
                },
            ]

        optimizer = self.optimizer(parameters, **optimizer_kwargs)

        if self.scheduler:
            warmup_batches = None
            if "warmup_batches" in self.scheduler_kwargs:
                warmup_batches = self.scheduler_kwargs.pop("warmup_batches")
            scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
            if warmup_batches:
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            optimizer, start_factor=0.01, total_iters=warmup_batches
                        ),
                        scheduler,
                    ],
                    milestones=[warmup_batches],
                )
            scheduler_config = {"scheduler": scheduler, "interval": "step"}
            scheduler_config["monitor"] = "loss"
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        return optimizer

    def on_validation_start(self) -> None:
        for head in self.model.heads:
            head: Head
            if hasattr(head, "on_validation_start"):
                head.on_validation_start()

    def on_validation_epoch_end(self) -> None:
        for head_idx, head in enumerate(self.model.heads):
            head: Head
            if not hasattr(head, "on_validation_end"):
                continue
            val_metrics = {
                f"{head_idx}/valid/{k}": v for k, v in head.on_validation_end().items()
            }
            try:
                self.log_dict(val_metrics, sync_dist=True)
            except MisconfigurationException:
                pass
        for viz_batch_idx, viz_batch in enumerate(self.visualization_batches):
            visualize(
                model=self.model,
                configs=self.data_config,
                input=viz_batch["input"],
                targets=viz_batch["targets"],
                logger=self.logger,
                step=self.global_step,
                start_idx=viz_batch_idx * self.visualize_n_per_batch,
            )

        torch.cuda.empty_cache()

    def on_pretraining_start(self) -> None:
        for head in self.model.heads:
            head: Head
            if hasattr(head, "on_pretraining_start"):
                head.on_pretraining_start()

    def on_pretraining_end(self) -> None:
        for head in self.model.heads:
            head: Head
            if hasattr(head, "on_pretraining_end"):
                head.on_pretraining_end()

from typing import Final, List, Dict, Type

from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor
import torch
import torchvision

from sihl.layers import AntialiasedDownscaler, Normalize
from sihl.utils import recursive_setattr, recursive_getattr, interpolate


TORCHVISION_MODEL_LEVELS: Final[Dict[str, List[str]]] = {
    "convnext_base": [f"features.{idx}" for idx in [0, 1, 3, 5, 7]],
    "convnext_large": [f"features.{idx}" for idx in [0, 1, 3, 5, 7]],
    "convnext_small": [f"features.{idx}" for idx in [0, 1, 3, 5, 7]],
    "convnext_tiny": [f"features.{idx}" for idx in [0, 1, 3, 5, 7]],
    "densenet121": ["features.relu0"]
    + [f"features.denseblock{_}" for _ in [1, 2, 3, 4]],
    "densenet161": ["features.relu0"]
    + [f"features.denseblock{_}" for _ in [1, 2, 3, 4]],
    "densenet169": ["features.relu0"]
    + [f"features.denseblock{_}" for _ in [1, 2, 3, 4]],
    "efficientnet_b0": [f"features.{_}" for _ in [1, 2, 3, 5, 8]],
    "efficientnet_b1": [f"features.{_}" for _ in [1, 2, 3, 5, 8]],
    "efficientnet_b2": [f"features.{_}" for _ in [1, 2, 3, 5, 8]],
    "efficientnet_b3": [f"features.{_}" for _ in [1, 2, 3, 5, 8]],
    "efficientnet_b4": [f"features.{_}" for _ in [1, 2, 3, 5, 8]],
    "efficientnet_b5": [f"features.{_}" for _ in [1, 2, 3, 5, 8]],
    "efficientnet_b6": [f"features.{_}" for _ in [1, 2, 3, 5, 8]],
    "efficientnet_b7": [f"features.{_}" for _ in [1, 2, 3, 5, 8]],
    "efficientnet_v2_l": [f"features.{_}" for _ in [1, 2, 3, 5, 8]],
    "efficientnet_v2_m": [f"features.{_}" for _ in [1, 2, 3, 5, 8]],
    "efficientnet_v2_s": [f"features.{_}" for _ in [1, 2, 3, 5, 7]],
    "mnasnet0_5": [f"layers.{_}" for _ in [7, 8, 9, 11, 16]],
    "mnasnet0_75": [f"layers.{_}" for _ in [7, 8, 9, 11, 16]],
    "mnasnet1_0": [f"layers.{_}" for _ in [7, 8, 9, 11, 16]],
    "mnasnet1_3": [f"layers.{_}" for _ in [7, 8, 9, 11, 16]],
    "mobilenet_v2": [f"features.{_}" for _ in [1, 3, 6, 13, 18]],
    "mobilenet_v3_large": [f"features.{_}" for _ in [1, 3, 6, 12, 16]],
    "mobilenet_v3_small": [f"features.{_}" for _ in [0, 1, 3, 8, 12]],
    "resnet101": ["relu"] + [f"layer{_}" for _ in [1, 2, 3, 4]],
    "resnet152": ["relu"] + [f"layer{_}" for _ in [1, 2, 3, 4]],
    "resnet18": ["relu"] + [f"layer{_}" for _ in [1, 2, 3, 4]],
    "resnet34": ["relu"] + [f"layer{_}" for _ in [1, 2, 3, 4]],
    "resnet50": ["relu"] + [f"layer{_}" for _ in [1, 2, 3, 4]],
    "resnext101_32x8d": ["relu"] + [f"layer{_}" for _ in [1, 2, 3, 4]],
    "resnext101_64x4d": ["relu"] + [f"layer{_}" for _ in [1, 2, 3, 4]],
    "resnext50_32x4d": ["relu"] + [f"layer{_}" for _ in [1, 2, 3, 4]],
    "shufflenet_v2_x0_5": ["conv1", "maxpool", "stage2", "stage3", "conv5"],
    "shufflenet_v2_x1_0": ["conv1", "maxpool", "stage2", "stage3", "conv5"],
    "shufflenet_v2_x1_5": ["conv1", "maxpool", "stage2", "stage3", "conv5"],
    "shufflenet_v2_x2_0": ["conv1", "maxpool", "stage2", "stage3", "conv5"],
    "wide_resnet50_2": ["relu"] + [f"layer{_}" for _ in [1, 2, 3, 4]],
    "wide_resnet101_2": ["relu"] + [f"layer{_}" for _ in [1, 2, 3, 4]],
}
FIRST_LAYERS: Final[Dict[str, str]] = {
    "convnext_base": "features.0.0",
    "convnext_large": "features.0.0",
    "convnext_small": "features.0.0",
    "convnext_tiny": "features.0.0",
    "densenet121": "features.conv0",
    "densenet161": "features.conv0",
    "densenet169": "features.conv0",
    "efficientnet_b0": "features.0.0",
    "efficientnet_b1": "features.0.0",
    "efficientnet_b2": "features.0.0",
    "efficientnet_b3": "features.0.0",
    "efficientnet_b4": "features.0.0",
    "efficientnet_b5": "features.0.0",
    "efficientnet_b6": "features.0.0",
    "efficientnet_b7": "features.0.0",
    "efficientnet_v2_l": "features.0.0",
    "efficientnet_v2_m": "features.0.0",
    "efficientnet_v2_s": "features.0.0",
    "mnasnet0_5": "layers.0",
    "mnasnet0_75": "layers.0",
    "mnasnet1_0": "layers.0",
    "mnasnet1_3": "layers.0",
    "mobilenet_v2": "features.0.0",
    "mobilenet_v3_large": "features.0.0",
    "mobilenet_v3_small": "features.0.0",
    "resnet101": "conv1",
    "resnet152": "conv1",
    "resnet18": "conv1",
    "resnet34": "conv1",
    "resnet50": "conv1",
    "resnext101_32x8d": "conv1",
    "resnext101_64x4d": "conv1",
    "resnext50_32x4d": "conv1",
    "shufflenet_v2_x0_5": "conv1.0",
    "shufflenet_v2_x1_0": "conv1.0",
    "shufflenet_v2_x1_5": "conv1.0",
    "shufflenet_v2_x2_0": "conv1.0",
    "wide_resnet50_2": "conv1",
    "wide_resnet101_2": "conv1",
}
TORCHVISION_BACKBONE_NAMES = tuple(TORCHVISION_MODEL_LEVELS.keys())


class TorchvisionBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        input_channels: int = 3,
        top_level: int = 5,
        frozen_levels: int = 0,
        freeze_batchnorms: bool = False,
        downscaler: Type[nn.Module] = AntialiasedDownscaler,
    ) -> None:
        """
        Args:
            name (str): Architecture name (must be in TORCHVISION_BACKBONE_NAMES)
            pretrained (bool, optional): Use weights from Imagenet pretraining. Defaults to False.
            input_channels (int, optional): Number of input image channels. Defaults to 3.
            top_level (int, optional): Deepest level (with stride 2^level). Defaults to 5.
            frozen_levels (int, optional): How many levels to freeze (from level 0). Defaults to 0.
            freeze_batchnorms (bool, optional): Whether to freeze batchnorm running stats. Defaults to False.
            downscaler (Type[nn.Module], optional): Downscaler module used for levels above 5. Defaults to AntialiasedDownscaler.
        """
        super().__init__()
        self.name = name
        self.top_level = top_level
        try:
            level_names = TORCHVISION_MODEL_LEVELS[name][: self.top_level]
        except KeyError as error:
            raise ValueError(
                f"Architecture {name} is not supported. "
                f"Select from {TORCHVISION_BACKBONE_NAMES}"
            ) from error
        self.normalize = nn.Identity()
        if pretrained and input_channels == 3:
            self.normalize = Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.model = torchvision.models.get_model(
            name, weights=("DEFAULT" if pretrained else None)
        )
        self.model = create_feature_extractor(self.model, level_names)

        first_conv = recursive_getattr(self.model, FIRST_LAYERS[name])
        if input_channels != first_conv.in_channels:
            recursive_setattr(
                self.model,
                FIRST_LAYERS[name],
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=first_conv.out_channels,
                    kernel_size=first_conv.kernel_size,
                    stride=first_conv.stride,
                    padding=first_conv.padding,
                    dilation=first_conv.dilation,
                    groups=first_conv.groups,
                    bias=first_conv.bias,
                    padding_mode=first_conv.padding_mode,
                ),
            )

        # freeze modules in first `frozen_levels` levels
        if pretrained:
            freeze_levels(self.model, frozen_levels, level_names, freeze_batchnorms)

        min_size = 2 ** (self.top_level + 1)
        self.dummy_input = torch.zeros(1, input_channels, min_size, min_size)
        self.out_channels = [input_channels] + [
            _.shape[1] for _ in self.model(self.dummy_input).values()
        ]
        out_channels = self.out_channels[-1]
        extra_layers = range(self.top_level - 5)
        self.out_channels += [out_channels for _ in extra_layers]
        self.downscalers = nn.ModuleList(
            [AntialiasedDownscaler(out_channels, out_channels) for _ in extra_layers]
        )

    def forward(self, input: Tensor) -> List[Tensor]:
        assert input.shape[2] % 2**self.top_level == 0
        assert input.shape[3] % 2**self.top_level == 0
        x = self.normalize(input)
        outputs = list(self.model(x).values())
        outputs = [input] + [
            interpolate(output, size=(x.shape[2] // 2**level, x.shape[3] // 2**level))
            for output, level in zip(outputs, range(1, self.top_level + 1))
        ]
        for downscaler in self.downscalers:
            outputs.append(downscaler(outputs[-1]))
        return outputs


def freeze_levels(
    model: nn.Module,
    num_levels: int,
    level_names: List[str],
    freeze_batchnorms: bool = False,
) -> None:
    """freeze modules in first `num_levels` levels"""
    if num_levels < 0:
        for param in model.parameters():
            param.requires_grad_(False)
    elif num_levels > 0:
        num_levels = min(num_levels, len(level_names))
        last_level_to_freeze = level_names[num_levels - 1]
        last_level_reached = False
        freezing = True
        for module_name, _ in model.named_modules():
            if module_name == "":
                continue
            if module_name == last_level_to_freeze:
                last_level_reached = True
            if last_level_reached and last_level_to_freeze not in module_name:
                freezing = False
            for param in model.get_submodule(module_name).parameters():
                param.requires_grad_(not freezing)
    else:
        for param in model.parameters():
            param.requires_grad_(True)

    if freeze_batchnorms:
        for layer in model.modules():
            if isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                layer.eval()

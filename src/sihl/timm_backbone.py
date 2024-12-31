from typing import List, Type

from math import log

from torch import Tensor, nn
import timm
import torch

from sihl.layers import AntialiasedDownscaler, Normalize
from sihl.utils import interpolate
from sihl.torchvision_backbone import freeze_levels

TIMM_BACKBONE_NAMES = (
    "convnext_atto",
    "convnext_base",
    "convnext_femto",
    "convnext_large",
    "convnext_nano",
    "convnext_pico",
    "convnext_small",
    "convnext_tiny",
    "convnext_xlarge",
    "convnext_xxlarge",
    "convnextv2_atto",
    "convnextv2_base",
    "convnextv2_femto",
    # "convnextv2_huge",
    "convnextv2_large",
    "convnextv2_nano",
    "convnextv2_pico",
    # "convnextv2_small",  # missing weights?!
    "convnextv2_tiny",
    "dla34",
    "dla60",
    "dla102",
    "dla169",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    # "efficientnet_b6",
    # "efficientnet_b7",
    # "efficientnet_b8",
    "efficientnet_lite0",
    # "efficientnet_lite1",
    # "efficientnet_lite2",
    # "efficientnet_lite3",
    # "efficientnet_lite4",
    # "efficientnetv2_l",
    # "efficientnetv2_m",
    # "efficientnetv2_s",
    # "efficientnetv2_xl",
    "hrnet_w18",
    "hrnet_w30",
    "hrnet_w32",
    "hrnet_w40",
    "hrnet_w44",
    "hrnet_w48",
    "hrnet_w64",
    # "mobilenetv2_035",
    "mobilenetv2_050",
    # "mobilenetv2_075",
    "mobilenetv2_100",
    "mobilenetv2_140",
    # "mobilenetv3_large_075",
    "mobilenetv3_large_100",
    "mobilenetv3_small_050",
    "mobilenetv3_small_075",
    "mobilenetv3_small_100",
    "mobilenetv4_conv_large",
    "mobilenetv4_conv_medium",
    "mobilenetv4_conv_small",
    "mobilenetv4_hybrid_large",
    # "mobilenetv4_hybrid_large_075",
    "mobilenetv4_hybrid_medium",
    # "mobilenetv4_hybrid_medium_075",
    "resnet18",
    "resnet26",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    # "resnet200",
    "resnetv2_50",
    "resnetv2_101",
    # "resnetv2_152",
)


class TimmBackbone(nn.Module):
    """https://github.com/huggingface/pytorch-image-models"""

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
            name (str): Architecture name (must be in TIMM_BACKBONE_NAMES)
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
            self.model = timm.create_model(
                self.name,
                features_only=True,
                scriptable=True,
                exportable=True,
                pretrained=pretrained,
                in_chans=input_channels,
            )
        except KeyError as error:
            raise ValueError(
                f"Architecture {name} is not supported. "
                f"Select from {TIMM_BACKBONE_NAMES}"
            ) from error
        except RuntimeError as error:
            raise ValueError("Failed to create TimmBackbone") from error
        reds = self.model.feature_info.reduction()
        assert all(log(reduction, 2).is_integer() for reduction in reds)

        self.normalize = nn.Identity()
        if pretrained and input_channels == 3:
            self.normalize = Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        # Some models don't have a level 1 (stide 2), so we manually "add" it
        # FIXME: this is a hack
        if 2 not in reds:
            old_forward = self.model.forward

            def new_forward(self, x):
                size = (x.shape[2] // 2, x.shape[3] // 2)
                return [interpolate(x, size=size)] + old_forward(x)

            self.model.forward = new_forward.__get__(self.model)

        # freeze modules in first `frozen_levels` levels
        if pretrained:
            level_names = self.model.feature_info.module_name()
            # FIXME: this is a hack
            if "convnext" in name:
                level_names = [_.replace(".", "_") for _ in level_names]
            freeze_levels(self.model, frozen_levels, level_names, freeze_batchnorms)

        min_size = 2 ** (self.top_level + 1)
        self.dummy_input = torch.zeros(1, input_channels, min_size, min_size)
        self.out_channels = [input_channels] + [
            _.shape[1] for _ in self.model.forward(self.dummy_input)
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
        outputs = self.model.forward(x)
        outputs = [input] + [
            interpolate(output, size=(x.shape[2] // 2**level, x.shape[3] // 2**level))
            for output, level in zip(outputs, range(1, self.top_level + 1))
        ]
        for downscaler in self.downscalers:
            outputs.append(downscaler(outputs[-1]))
        return outputs

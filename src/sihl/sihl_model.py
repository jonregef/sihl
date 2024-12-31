from typing import List, Optional

from torch import Tensor, nn


class SihlModel(nn.Module):
    """A SihlModel consists of a backbone, zero or one neck, and one or more head(s)."""

    def __init__(
        self, backbone: nn.Module, neck: Optional[nn.Module], heads: List[nn.Module]
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = nn.ModuleList(heads)

    def extract_features(self, input: Tensor) -> List[Tensor]:
        x = self.backbone(input)
        if self.neck is not None:
            return self.neck(x)
        return x

    def forward(self, input: Tensor) -> List[Tensor]:
        x = self.extract_features(input)
        return [head(x) for head in self.heads]

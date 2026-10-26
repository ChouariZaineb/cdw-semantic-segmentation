"""FPN-based semantic segmentation model definition."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FPNSegmenter(nn.Module):
    """
    Semantic segmentation model:
      image -> ResNet backbone -> FPN features -> fuse -> segmentation logits

    backbone_name accepte: "resnet50", "resnet101", "resnet152"
    Les canaux FPN restent à 256 pour tous les backbones.
    """

    def __init__(
        self,
        num_classes: int = 11,
        backbone_name: str = "resnet101",
        pretrained: bool = True,
    ):
        super().__init__()

        self.backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            weights="DEFAULT" if pretrained else None,
        )

        fpn_out = 256

        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_out, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]

        feats: Dict[str, torch.Tensor] = self.backbone(x)

        p2 = feats["0"]
        target_hw = p2.shape[-2:]

        fused = torch.zeros_like(p2)
        for k, f in feats.items():
            if f.shape[-2:] != target_hw:
                f = F.interpolate(f, size=target_hw, mode="nearest")
            fused = fused + f

        fused = self.fuse(fused)
        logits = self.classifier(fused)

        if logits.shape[-2:] != input_hw:
            logits = F.interpolate(
                logits, size=input_hw, mode="bilinear", align_corners=False
            )

        return logits
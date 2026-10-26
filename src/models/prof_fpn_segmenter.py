from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ----------------------
# Backbone (Prof)
# ----------------------
class ResNet50_Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # C2: 256
        self.layer2 = resnet.layer2  # C3: 512
        self.layer3 = resnet.layer3  # C4: 1024
        self.layer4 = resnet.layer4  # C5: 2048

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


# ----------------------
# FPN (Prof) - segmentation-friendly
# ----------------------
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.smooth_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])

    def forward(self, features):
        c2, c3, c4, c5 = features

        p5 = self.lateral_convs[3](c5)

        # use size=... to avoid off-by-1 issues
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        p5 = self.smooth_convs[3](p5)
        p4 = self.smooth_convs[2](p4)
        p3 = self.smooth_convs[1](p3)
        p2 = self.smooth_convs[0](p2)

        return p2, p3, p4, p5


# ----------------------
# Segmentation Head
# ----------------------
class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 11):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ----------------------
# Full model: ResNet50 + FPN + Segmentation Head
# ----------------------
class ResNet50_FPN_Segmenter(nn.Module):
    def __init__(self, num_classes: int = 11, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNet50_Backbone(pretrained=pretrained)
        self.fpn = FPN([256, 512, 1024, 2048], 256)
        self.head = SegmentationHead(in_channels=256, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        features = self.backbone(x)
        p2, p3, p4, p5 = self.fpn(features)

        logits = self.head(p2)  # highest resolution pyramid level
        logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return logits
from __future__ import annotations
"""Build segmentation models from architecture aliases."""

from src.models.fpn_segmenter import FPNSegmenter

_ARCH_BACKBONE = {
    "student":    "resnet50",
    "ours":       "resnet50",
    "fpn":        "resnet50",
    "fpn101":     "resnet101",
    "student101": "resnet101",
    "fpn152":     "resnet152",
    "student152": "resnet152",
}


def build_model(arch: str, num_classes: int = 11):
    """
    Construit le modèle de segmentation selon l'architecture demandée.

    Args:
        arch:        Nom de l'architecture.
                       "student" / "ours" / "fpn"     -> ResNet50  + FPN
                       "fpn101"  / "student101"        -> ResNet101 + FPN
                       "fpn152"  / "student152"        -> ResNet152 + FPN
        num_classes: Nombre de classes (défaut: 11).

    Returns:
        FPNSegmenter initialisé avec les poids ImageNet.
    """
    arch_key = arch.lower().strip()

    if arch_key not in _ARCH_BACKBONE:
        raise ValueError(
            f"Unknown arch={arch!r}. "
            f"Valid options: {sorted(_ARCH_BACKBONE.keys())}"
        )

    backbone_name = _ARCH_BACKBONE[arch_key]
    print(f"[build_model] arch={arch_key} → backbone={backbone_name}")

    return FPNSegmenter(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=True,
    )
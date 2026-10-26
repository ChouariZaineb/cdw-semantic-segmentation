from __future__ import annotations
"""Backbone freezing and unfreezing helpers for multiphase fine-tuning."""

import torch.nn as nn


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _get_resnet_body(model: nn.Module) -> nn.Module:
    """
    Returns the ResNet module containing conv1/bn1/layer1..layer4.
    For FPNSegmenter, model.backbone.body is the ResNet backbone body.
    """
    if hasattr(model, "backbone") and hasattr(model.backbone, "body"):
        return model.backbone.body
    raise AttributeError("Model has no recognizable backbone.")


def freeze_backbone_all(model: nn.Module) -> None:
    body = _get_resnet_body(model)
    set_requires_grad(body, False)


def unfreeze_backbone_c4_c5(model: nn.Module) -> None:
    body = _get_resnet_body(model)
    set_requires_grad(body, False)
    set_requires_grad(body.layer3, True)
    set_requires_grad(body.layer4, True)


def unfreeze_backbone_all(model: nn.Module) -> None:
    body = _get_resnet_body(model)
    set_requires_grad(body, True)
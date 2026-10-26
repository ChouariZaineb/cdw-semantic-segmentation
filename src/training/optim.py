from __future__ import annotations
"""Optimizer builder with separate learning rates for backbone and head."""

import torch
import torch.nn as nn


def make_optimizer_two_lrs(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_backbone_body = name.startswith("backbone.body.")
        if is_backbone_body:
            backbone_params.append(p)
        else:
            head_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
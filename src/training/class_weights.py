from __future__ import annotations
"""Class reweighting utilities for imbalanced segmentation labels."""

import torch


def median_frequency_balancing_from_counts(
    class_pixels: torch.Tensor,
    ignore_index: int = 0,
    clip_max: float | None = 10.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    class_pixels: tensor [C] of pixel counts for classes (including ignore_index).
    Returns weights [C] where weights[ignore_index]=0.
    """
    counts = class_pixels.to(torch.float64).clone()
    counts[ignore_index] = 0.0

    total_labeled = counts.sum().clamp_min(1.0)
    freqs = counts / total_labeled

    valid = torch.ones_like(freqs, dtype=torch.bool)
    valid[ignore_index] = False
    valid &= freqs > 0

    med = torch.median(freqs[valid]) if torch.any(valid) else torch.tensor(1.0, dtype=torch.float64)

    weights = torch.zeros_like(freqs)
    weights[valid] = med / (freqs[valid] + eps)

    if clip_max is not None:
        weights = torch.clamp(weights, max=float(clip_max))

    weights[ignore_index] = 0.0
    return weights.to(torch.float32)
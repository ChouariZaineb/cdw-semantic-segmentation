from __future__ import annotations
"""Parser for pixel statistics used to compute class weighting."""

from dataclasses import dataclass
from pathlib import Path
import re
import torch


@dataclass
class PixelStats:
    total_pixels: int
    ignore_pixels: int
    labeled_pixels: int
    class_pixels: torch.Tensor  # shape [num_classes], includes class 0


_LINE_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*=\s*([0-9]+)\s*$")


def load_pixel_stats_txt(path: str | Path, num_classes: int = 11, ignore_index: int = 0) -> PixelStats:
    """
    Parse a pixel_stats.txt file like:
      total_pixels=...
      ignore_pixels=...
      labeled_pixels=...
      class_1_pixels=...
      ...
      class_10_pixels=...

    Returns class_pixels with length num_classes where class_pixels[0]=ignore_pixels.
    """
    path = Path(path)
    data: dict[str, int] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            raise ValueError(f"Unrecognized line in {path}: {line!r}")
        key, val = m.group(1), int(m.group(2))
        data[key] = val

    total_pixels = data.get("total_pixels", 0)
    ignore_pixels = data.get("ignore_pixels", 0)
    labeled_pixels = data.get("labeled_pixels", 0)

    class_pixels = torch.zeros(num_classes, dtype=torch.int64)
    class_pixels[ignore_index] = ignore_pixels

    # Fill class_1..class_(num_classes-1)
    for c in range(1, num_classes):
        k = f"class_{c}_pixels"
        if k in data:
            class_pixels[c] = int(data[k])

    return PixelStats(
        total_pixels=int(total_pixels),
        ignore_pixels=int(ignore_pixels),
        labeled_pixels=int(labeled_pixels),
        class_pixels=class_pixels,
    )
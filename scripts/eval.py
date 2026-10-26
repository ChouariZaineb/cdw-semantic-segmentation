from __future__ import annotations
"""Checkpoint evaluation script that exports IoU metrics and confusion matrix."""

from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader

from scripts.utils import dump_json
from src.data.cdw_dataset import CDWSegmentationDataset
from src.data.transforms import SegmentationBasicTransform, SegmentationTransformConfig
from src.models.build_model import build_model
from src.training.metrics import compute_iou_from_confmat, update_confusion_matrix


def resolve_device(requested_device: str) -> torch.device:
    requested_device = requested_device.lower()
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if requested_device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested_device}. Use auto/cuda/cpu")


@torch.no_grad()
def evaluate_checkpoint(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> dict:
    model.eval()
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, masks, _ids in dl:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        masks = masks.to(device, non_blocking=(device.type == "cuda"))

        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()
        masks = masks.cpu()

        confmat = update_confusion_matrix(confmat, preds, masks, num_classes, ignore_index)

    m = compute_iou_from_confmat(confmat, ignore_index=ignore_index)
    return {
        "pixel_acc": float(m["pixel_acc"].item()),
        "miou": float(m["miou"].item()),
        "per_class_iou": [float(v) for v in m["per_class_iou"].tolist()],
        "valid_pixel_count": int(m["valid_pixel_count"].item()),
        "confmat": confmat.tolist(),
    }


def print_report(metrics: dict, ignore_index: int) -> None:
    print(f"pixel_acc={metrics['pixel_acc']:.4f} miou={metrics['miou']:.4f}")
    print("per_class_iou:")
    for cls, val in enumerate(metrics["per_class_iou"]):
        if cls == ignore_index:
            continue
        print(f"  class_{cls}: {val:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arch",
        type=str,
        default="student",
        choices=["student", "fpn", "fpn101", "student101", "fpn152", "student152"],
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--split-file", type=str, required=True)
    ap.add_argument("--out-json", type=str, default="results/eval_results.json")

    ap.add_argument("--num-classes", type=int, default=11)
    ap.add_argument("--ignore-index", type=int, default=0)
    ap.add_argument("--image-size", type=int, nargs=2, default=[768, 768])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    device = resolve_device(args.device)

    tfm = SegmentationBasicTransform(
        SegmentationTransformConfig(size=tuple(args.image_size), keep_aspect_ratio=True)
    )
    ds = CDWSegmentationDataset(
        dataset_root=Path(args.dataset_root),
        split_file=Path(args.split_file),
        transform=tfm,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(arch=args.arch, num_classes=args.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    metrics = evaluate_checkpoint(
        model=model,
        dl=dl,
        device=device,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": args.arch,
        "checkpoint": str(Path(args.ckpt)),
        "dataset_root": str(Path(args.dataset_root)),
        "split_file": str(Path(args.split_file)),
        "device": str(device),
        "image_size": list(args.image_size),
        "metrics": metrics,
    }
    dump_json(out_json, payload)

    print_report(metrics, ignore_index=args.ignore_index)
    print(f"Saved eval report: {out_json}")


if __name__ == "__main__":
    main()

"""Main multiphase training entrypoint — RUN 3.

Changements vs Run 2 :
- Focal Loss supprimée → Cross-Entropy pondérée
- RandomResizedCrop supprimé dans transforms
- Epochs augmentées : P1=5, P2=12, P3=35
- Patience augmentée : 5 → 8
- Oversampling conservé (C5/C10 ×6, C8 ×4)
- Augmentation légère conservée
"""

from __future__ import annotations

from pathlib import Path
import argparse
import time

import torch
from torch.utils.data import DataLoader

from scripts.utils import append_jsonl, deep_update, dump_json, dump_yaml, load_yaml_config, make_experiment_dir, set_global_seed
from src.data.cdw_dataset import CDWSegmentationDataset
from src.data.transforms import (
    SegmentationBasicTransform,
    SegmentationAugmentedTransform,
    SegmentationTransformConfig,
)
from src.models.build_model import build_model
from src.training.class_weights import median_frequency_balancing_from_counts
from src.training.freeze import freeze_backbone_all, unfreeze_backbone_all, unfreeze_backbone_c4_c5
from src.training.losses import make_weighted_ce_loss   # ← Cross-Entropy (pas Focal)
from src.training.optim import make_optimizer_two_lrs
from src.training.pixel_stats import load_pixel_stats_txt


def train_one_epoch(model, dl, criterion, optimizer, device, grad_accum_steps=1, max_batches=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    step = 0

    for step, (images, masks, ids) in enumerate(dl, start=1):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        masks  = masks.to(device,  non_blocking=(device.type == "cuda"))

        logits = model(images)
        loss   = criterion(logits, masks) / grad_accum_steps
        loss.backward()

        if step % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running += float(loss.item()) * grad_accum_steps

        if step % 20 == 0:
            print(f"  step={step:04d} loss={float(loss.item()) * grad_accum_steps:.4f} avg={running / step:.4f}")

        if max_batches is not None and step >= max_batches:
            break

    if step > 0 and step % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if step == 0:
        return 0.0

    return running / step


@torch.no_grad()
def evaluate_metrics(model, dl, device, num_classes=11, ignore_index=0):
    from src.training.metrics import update_confusion_matrix, compute_iou_from_confmat

    model.eval()
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, masks, ids in dl:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        masks  = masks.to(device,  non_blocking=(device.type == "cuda"))

        logits = model(images)
        preds  = torch.argmax(logits, dim=1).cpu()
        masks  = masks.cpu()

        confmat = update_confusion_matrix(confmat, preds, masks, num_classes, ignore_index)

    m = compute_iou_from_confmat(confmat, ignore_index)
    return {
        "pixel_acc":       float(m["pixel_acc"].item()),
        "miou":            float(m["miou"].item()),
        "per_class_iou":   [float(v) for v in m["per_class_iou"].tolist()],
        "valid_pixel_count": int(m["valid_pixel_count"].item()),
    }


def evaluate_and_print(model, dl, device, split_name, num_classes=11, ignore_index=0):
    m = evaluate_metrics(model, dl, device, num_classes, ignore_index)
    print(f"  -> {split_name}: acc={m['pixel_acc']:.4f} miou={m['miou']:.4f}")
    return m


def _worst_class_summary(per_class_iou: list[float], ignore_index: int, k: int = 3) -> str:
    pairs = [(c, s) for c, s in enumerate(per_class_iou) if c != ignore_index]
    pairs.sort(key=lambda x: x[1])
    return ", ".join(f"c{c}:{v:.3f}" for c, v in pairs[:k])


def resolve_device(requested_device: str) -> torch.device:
    requested_device = requested_device.lower()
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA non disponible → CPU.")
        return torch.device("cpu")
    if requested_device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested_device}")


def _default_config() -> dict:
    return {
        "experiment": {
            "name": "run3_aug_light_oversample_ce",
            "seed": 42,                  # ← même seed que Run 1 et Run 2
        },
        "model": {
            "arch": "student",           # ← ResNet50 inchangé
        },
        "dataset": {
            "dataset_root": None,
            "splits_dir":   None,
            "num_classes":  11,
            "ignore_index": 0,
            "image_size":   [768, 768],
        },
        "training": {
            "batch_size":           2,
            "num_workers":          2,
            "grad_accum_steps":     1,
            "weight_decay":         1e-4,
            "patience":             8,   # ← était 5, plus de patience
            "class_weight_clip_max": 10.0,
            "oversample_rare":      True,  # ← conservé
            "epochs": {
                "phase1": 5,             # ← était 3
                "phase2": 12,            # ← était 8
                "phase3": 35,            # ← était 20
            },
            "lrs": {
                "phase1": {"head": 1e-4, "backbone": 0.0},
                "phase2": {"head": 1e-4, "backbone": 1e-5},
                "phase3": {"head": 5e-5, "backbone": 5e-6},
            },
        },
        "output": {
            "out_dir": "runs",
        },
    }


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    out = deep_update({}, cfg)
    if args.arch is not None:
        out["model"]["arch"] = args.arch
    if args.dataset_root is not None:
        out["dataset"]["dataset_root"] = args.dataset_root
    if args.splits_dir is not None:
        out["dataset"]["splits_dir"] = args.splits_dir
    if args.out_dir is not None:
        out["output"]["out_dir"] = args.out_dir
    if args.batch_size is not None:
        out["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        out["training"]["num_workers"] = args.num_workers
    if args.grad_accum_steps is not None:
        out["training"]["grad_accum_steps"] = args.grad_accum_steps
    if args.seed is not None:
        out["experiment"]["seed"] = args.seed
    if args.exp_name is not None:
        out["experiment"]["name"] = args.exp_name
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           type=str,   default=None)
    parser.add_argument("--exp-name",         type=str,   default=None)
    parser.add_argument("--seed",             type=int,   default=None)
    parser.add_argument("--arch",             type=str,   default=None,
                        choices=["student", "ours", "fpn",
                                 "fpn101", "student101",
                                 "fpn152", "student152"])
    parser.add_argument("--device",           type=str,   default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dataset-root",     type=str,   default=None)
    parser.add_argument("--splits-dir",       type=str,   default=None)
    parser.add_argument("--out-dir",          type=str,   default=None)
    parser.add_argument("--batch-size",       type=int,   default=None)
    parser.add_argument("--num-workers",      type=int,   default=None)
    parser.add_argument("--grad-accum-steps", type=int,   default=None)

    args = parser.parse_args()

    cfg = _default_config()
    cfg = deep_update(cfg, load_yaml_config(args.config))
    cfg = _apply_cli_overrides(cfg, args)

    if not cfg["dataset"]["dataset_root"]:
        raise ValueError("dataset.dataset_root is required")
    if not cfg["dataset"]["splits_dir"]:
        raise ValueError("dataset.splits_dir is required")

    dataset_root     = Path(cfg["dataset"]["dataset_root"])
    splits_dir       = Path(cfg["dataset"]["splits_dir"])
    out_root         = Path(cfg["output"]["out_dir"])
    train_split      = splits_dir / "train.txt"
    val_split        = splits_dir / "val.txt"
    check_split      = splits_dir / "check.txt"
    pixel_stats_path = splits_dir / "pixel_stats.txt"

    device = resolve_device(args.device)
    set_global_seed(int(cfg["experiment"]["seed"]))

    num_classes           = int(cfg["dataset"]["num_classes"])
    ignore_index          = int(cfg["dataset"]["ignore_index"])
    image_size            = tuple(cfg["dataset"]["image_size"])
    batch_size            = int(cfg["training"]["batch_size"])
    num_workers           = int(cfg["training"]["num_workers"])
    grad_accum_steps      = int(cfg["training"]["grad_accum_steps"])
    epochs_p1             = int(cfg["training"]["epochs"]["phase1"])
    epochs_p2             = int(cfg["training"]["epochs"]["phase2"])
    epochs_p3             = int(cfg["training"]["epochs"]["phase3"])
    lr_head_p1            = float(cfg["training"]["lrs"]["phase1"]["head"])
    lr_backbone_p1        = float(cfg["training"]["lrs"]["phase1"]["backbone"])
    lr_head_p2            = float(cfg["training"]["lrs"]["phase2"]["head"])
    lr_backbone_p2        = float(cfg["training"]["lrs"]["phase2"]["backbone"])
    lr_head_p3            = float(cfg["training"]["lrs"]["phase3"]["head"])
    lr_backbone_p3        = float(cfg["training"]["lrs"]["phase3"]["backbone"])
    weight_decay          = float(cfg["training"]["weight_decay"])
    patience              = int(cfg["training"]["patience"])
    class_weight_clip_max = float(cfg["training"]["class_weight_clip_max"])
    oversample_rare       = bool(cfg["training"]["oversample_rare"])
    arch                  = str(cfg["model"]["arch"])

    print("=" * 55)
    print("RUN 3 — AUGMENTATION LÉGÈRE + OVERSAMPLING + CE")
    print("=" * 55)
    print("Architecture:    ", arch)
    print("Augmentation:    ", "OUI (légère, sans RandomResizedCrop)")
    print("Oversampling:    ", "OUI (C5/C10 ×6, C8 ×4)")
    print("Loss:            ", "Cross-Entropy pondérée")
    print("Epochs:          ", f"P1={epochs_p1} P2={epochs_p2} P3={epochs_p3}")
    print("Patience:        ", patience)
    print("Seed:            ", cfg["experiment"]["seed"], " ← identique Run 1/2")
    print("=" * 55)

    train_tfm = SegmentationAugmentedTransform(
        SegmentationTransformConfig(size=image_size, keep_aspect_ratio=True)
    )
    val_tfm = SegmentationBasicTransform(
        SegmentationTransformConfig(size=image_size, keep_aspect_ratio=True)
    )

    train_ds = CDWSegmentationDataset(
        dataset_root=dataset_root,
        split_file=train_split,
        transform=train_tfm,
        oversample_rare=oversample_rare,
    )
    val_ds = CDWSegmentationDataset(
        dataset_root=dataset_root,
        split_file=val_split,
        transform=val_tfm,
        oversample_rare=False,
    )

    has_check = check_split.exists()
    check_ds  = None
    if has_check:
        check_ds = CDWSegmentationDataset(
            dataset_root=dataset_root,
            split_file=check_split,
            transform=val_tfm,
            oversample_rare=False,
        )

    pin_memory = device.type == "cuda"

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)
    check_dl = None
    if has_check and check_ds is not None:
        check_dl = DataLoader(check_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    model = build_model(arch=arch, num_classes=num_classes).to(device)

    stats   = load_pixel_stats_txt(pixel_stats_path, num_classes=num_classes,
                                   ignore_index=ignore_index)
    weights = median_frequency_balancing_from_counts(
        stats.class_pixels, ignore_index=ignore_index,
        clip_max=class_weight_clip_max,
    )

    # ── Cross-Entropy pondérée — plus stable que Focal sur petit dataset ──────
    criterion = make_weighted_ce_loss(
        num_classes=num_classes,
        class_weights=weights.tolist(),
        ignore_index=ignore_index,
    ).to(device)

    exp_name = str(cfg["experiment"].get("name") or f"run3_{arch}")
    out_dir  = make_experiment_dir(
        out_root, exp_name=f"{exp_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    metrics_path = out_dir / "metrics.jsonl"
    dump_yaml(out_dir / "config_snapshot.yaml", cfg)

    def save_ckpt(tag: str, epoch: int | None = None, val_miou: float | None = None):
        torch.save({
            "model_state":   model.state_dict(),
            "tag":           tag,
            "arch":          arch,
            "image_size":    image_size,
            "ignore_index":  ignore_index,
            "num_classes":   num_classes,
            "seed":          int(cfg["experiment"]["seed"]),
            "class_weights": [float(v) for v in weights.tolist()],
            "epoch":         epoch,
            "val_miou":      val_miou,
            "config":        cfg,
        }, out_dir / f"model_{tag}.pt")
        print("Saved:", out_dir / f"model_{tag}.pt")

    # ── PHASE 1 ───────────────────────────────────────────────────────────────
    print(f"\nPHASE 1: Freeze backbone ({epochs_p1} epochs)")
    freeze_backbone_all(model)
    optimizer = make_optimizer_two_lrs(model, lr_backbone=lr_backbone_p1,
                                       lr_head=lr_head_p1, weight_decay=weight_decay)
    best_miou_p1 = -1.0
    for ep in range(1, epochs_p1 + 1):
        t0      = time.time()
        tr_loss = train_one_epoch(model, train_dl, criterion, optimizer, device, grad_accum_steps)
        val     = evaluate_metrics(model, val_dl, device, num_classes, ignore_index)
        print(f"[P1][ep {ep}/{epochs_p1}] loss={tr_loss:.4f} "
              f"acc={val['pixel_acc']:.4f} miou={val['miou']:.4f} "
              f"time={time.time()-t0:.1f}s")
        print(f"    worst: {_worst_class_summary(val['per_class_iou'], ignore_index)}")

        check = None
        if has_check and check_dl is not None:
            check = evaluate_and_print(model, check_dl, device, "check", num_classes, ignore_index)

        append_jsonl(metrics_path, {"phase": "p1", "epoch": ep, "train_loss": tr_loss,
                                    "val": val, "check": check, "elapsed_sec": time.time()-t0})
        if val["miou"] > best_miou_p1 + 1e-4:
            best_miou_p1 = val["miou"]
            save_ckpt(f"phase1_best_ep{ep}", epoch=ep, val_miou=val["miou"])

    save_ckpt("phase1_last")

    # ── PHASE 2 ───────────────────────────────────────────────────────────────
    print(f"\nPHASE 2: Unfreeze C4-C5 ({epochs_p2} epochs)")
    unfreeze_backbone_c4_c5(model)
    optimizer = make_optimizer_two_lrs(model, lr_backbone=lr_backbone_p2,
                                       lr_head=lr_head_p2, weight_decay=weight_decay)
    best_miou_p2 = -1.0
    for ep in range(1, epochs_p2 + 1):
        t0      = time.time()
        tr_loss = train_one_epoch(model, train_dl, criterion, optimizer, device, grad_accum_steps)
        val     = evaluate_metrics(model, val_dl, device, num_classes, ignore_index)
        print(f"[P2][ep {ep}/{epochs_p2}] loss={tr_loss:.4f} "
              f"acc={val['pixel_acc']:.4f} miou={val['miou']:.4f} "
              f"time={time.time()-t0:.1f}s")
        print(f"    worst: {_worst_class_summary(val['per_class_iou'], ignore_index)}")

        check = None
        if has_check and check_dl is not None:
            check = evaluate_and_print(model, check_dl, device, "check", num_classes, ignore_index)

        append_jsonl(metrics_path, {"phase": "p2", "epoch": ep, "train_loss": tr_loss,
                                    "val": val, "check": check, "elapsed_sec": time.time()-t0})
        if val["miou"] > best_miou_p2 + 1e-4:
            best_miou_p2 = val["miou"]
            save_ckpt(f"phase2_best_ep{ep}", epoch=ep, val_miou=val["miou"])

    save_ckpt("phase2_last")

    # ── PHASE 3 ───────────────────────────────────────────────────────────────
    print(f"\nPHASE 3: Unfreeze all ({epochs_p3} epochs max, patience={patience})")
    unfreeze_backbone_all(model)
    optimizer = make_optimizer_two_lrs(model, lr_backbone=lr_backbone_p3,
                                       lr_head=lr_head_p3, weight_decay=weight_decay)
    best_miou_p3 = -1.0
    bad = 0

    for ep in range(1, epochs_p3 + 1):
        t0      = time.time()
        tr_loss = train_one_epoch(model, train_dl, criterion, optimizer, device, grad_accum_steps)
        val     = evaluate_metrics(model, val_dl, device, num_classes, ignore_index)
        print(f"[P3][ep {ep}/{epochs_p3}] loss={tr_loss:.4f} "
              f"acc={val['pixel_acc']:.4f} miou={val['miou']:.4f} "
              f"time={time.time()-t0:.1f}s")
        print(f"    worst: {_worst_class_summary(val['per_class_iou'], ignore_index)}")

        check = None
        if has_check and check_dl is not None:
            check = evaluate_and_print(model, check_dl, device, "check", num_classes, ignore_index)

        append_jsonl(metrics_path, {"phase": "p3", "epoch": ep, "train_loss": tr_loss,
                                    "val": val, "check": check, "elapsed_sec": time.time()-t0})

        if val["miou"] > best_miou_p3 + 1e-4:
            best_miou_p3 = val["miou"]
            bad = 0
            save_ckpt(f"phase3_best_ep{ep}", epoch=ep, val_miou=val["miou"])
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping après {patience} epochs sans amélioration.")
                break

    save_ckpt("phase3_last")
    dump_json(out_dir / "summary.json", {
        "best_miou": {
            "phase1": best_miou_p1,
            "phase2": best_miou_p2,
            "phase3": best_miou_p3,
        },
        "run":            "run3_aug_light_oversample_ce",
        "arch":           arch,
        "augmentation":   "légère (sans RandomResizedCrop)",
        "oversampling":   "C5/C10 ×6, C8 ×4",
        "loss":           "weighted_cross_entropy",
        "epochs":         {"p1": epochs_p1, "p2": epochs_p2, "p3": epochs_p3},
        "patience":       patience,
        "seed":           int(cfg["experiment"]["seed"]),
        "metrics_path":   str(metrics_path),
    })

    print("\nDone — Run 3 terminé.")
    print(f"Artifacts saved in: {out_dir}")
    print(f"\n{'='*55}")
    print(f"TABLEAU DE COMPARAISON :")
    print(f"  Run 1 baseline  →  0.6028")
    print(f"  Run 2 amélioré  →  0.5790  (régression)")
    print(f"  Run 3           →  {best_miou_p3:.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()

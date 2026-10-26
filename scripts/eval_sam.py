from __future__ import annotations

import argparse
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

print("[BOOT] eval_sam.py loaded", flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.build_model import build_model
from src.training.metrics import update_confusion_matrix, compute_iou_from_confmat
from scripts.utils import dump_json

# ──────────────────────────────────────────────────────────────────────────────
NUM_CLASSES = 11
IGNORE_INDEX = 0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RGB_TO_CLASS = {
    (0, 0, 0): 0,
    (128, 0, 0): 1,
    (0, 128, 0): 2,
    (128, 128, 0): 3,
    (0, 0, 128): 4,
    (128, 0, 128): 5,
    (0, 128, 128): 6,
    (128, 128, 128): 7,
    (64, 0, 0): 8,
    (192, 0, 0): 9,
    (64, 128, 0): 10,
}

CLASS_NAMES = {
    1: "bin", 2: "concrete", 3: "fill_dirt",
    4: "timber", 5: "hard_plastic", 6: "soft_plastic",
    7: "steel", 8: "fabric", 9: "cardboard", 10: "plasterboard",
}


def rgb_mask_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    label = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    for rgb, cls in RGB_TO_CLASS.items():
        m = np.all(mask_rgb == np.array(rgb), axis=-1)
        label[m] = cls
    return label


def load_eval_ids(splits_dir: Path, n: int, seed: int = 42) -> list[str]:
    for fname in ["test.txt", "val.txt"]:
        f = splits_dir / fname
        if f.exists():
            if fname == "val.txt":
                print("[Info] test.txt absent -> fallback val.txt", flush=True)
            ids = [l.strip() for l in f.read_text().splitlines() if l.strip()]
            random.seed(seed)
            return random.sample(ids, min(n, len(ids)))
    raise FileNotFoundError(f"Aucun test.txt ou val.txt dans {splits_dir}")


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    print(f"[Model] loading: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "student")
    model = build_model(arch=arch, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval().to(device)
    print(f"[Model] arch={arch} | val_miou={ckpt.get('val_miou', 'N/A')}", flush=True)
    return model


def get_bbox_from_mask(binary_mask: np.ndarray):
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)


def infer_region_majority_class_and_score(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    region_size: int = 512,
    padding: int = 10,
    use_amp: bool = True,
):
    """
    Retourne:
      cls   : classe majoritaire (ignore 0)
      score : ratio majorit�� (0..1), utilisé pour résoudre overlap
    """
    bbox = get_bbox_from_mask(binary_mask)
    if bbox is None:
        return IGNORE_INDEX, 0.0

    rmin, rmax, cmin, cmax = bbox
    h, w = image_rgb.shape[:2]

    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)

    crop = image_rgb[rmin:rmax + 1, cmin:cmax + 1].copy()
    mask_crop = binary_mask[rmin:rmax + 1, cmin:cmax + 1]

    if crop.shape[0] < 5 or crop.shape[1] < 5 or mask_crop.sum() == 0:
        return IGNORE_INDEX, 0.0

    crop[~mask_crop] = 0

    x = Image.fromarray(crop.astype(np.uint8)).resize((region_size, region_size), Image.BILINEAR)
    x = TF.to_tensor(x)
    x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
        else:
            logits = model(x)

        # attendu: [1,C,H,W]
        if logits.ndim != 4:
            return IGNORE_INDEX, 0.0

        pred_map = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)

    # redimensionner masque région vers taille pred_map
    mask_t = torch.from_numpy(mask_crop.astype(np.float32))[None, None, ...]
    mask_rs = F.interpolate(mask_t, size=pred_map.shape, mode="nearest").squeeze().numpy().astype(bool)

    region_pixels = pred_map[mask_rs]
    if region_pixels.size == 0:
        return IGNORE_INDEX, 0.0

    valid = region_pixels[region_pixels != IGNORE_INDEX]
    if valid.size == 0:
        return IGNORE_INDEX, 0.0

    counts = np.bincount(valid, minlength=NUM_CLASSES)
    cls = int(np.argmax(counts))
    maj = int(counts[cls])
    score = float(maj / valid.size)

    return cls, score


def build_semantic_mask_vote_overlap_bestscore(
    image_rgb: np.ndarray,
    sam_masks: list[dict],
    model: torch.nn.Module,
    device: torch.device,
    max_masks_per_image: int,
    region_size: int,
    use_amp: bool = True,
):
    h, w = image_rgb.shape[:2]
    semantic = np.zeros((h, w), dtype=np.int64)
    best_score = np.full((h, w), -1.0, dtype=np.float32)

    masks = sorted(sam_masks, key=lambda x: x.get("area", 0), reverse=True)[:max_masks_per_image]

    kept, ignored = 0, 0
    for m in masks:
        binary = m["segmentation"].astype(bool)
        if binary.sum() == 0:
            ignored += 1
            continue

        cls, rscore = infer_region_majority_class_and_score(
            image_rgb=image_rgb,
            binary_mask=binary,
            model=model,
            device=device,
            region_size=region_size,
            padding=10,
            use_amp=use_amp,
        )

        if cls == IGNORE_INDEX:
            ignored += 1
            continue

        update = binary & (rscore > best_score)
        semantic[update] = cls
        best_score[update] = rscore
        kept += 1

    return semantic, kept, ignored


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    print("[BOOT] entering main()", flush=True)

    parser = argparse.ArgumentParser(description="Eval SAM2 + classifier (majority vote region, overlap=best-score)")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--sam2-ckpt", required=True)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--out-dir", default="runs/sam_eval_vote")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # SAM params
    parser.add_argument("--points-per-side", type=int, default=8)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.80)
    parser.add_argument("--stability-score-thresh", type=float, default=0.90)
    parser.add_argument("--min-mask-region-area", type=int, default=5000)
    parser.add_argument("--max-masks-per-image", type=int, default=80)

    # region classifier params
    parser.add_argument("--region-size", type=int, default=512, help="Resize des crops région (ex: 512 ou 756)")
    parser.add_argument("--no-amp", action="store_true", help="Désactiver autocast fp16 (par défaut activé sur CUDA)")

    args = parser.parse_args()
    print(f"[ARGS] {args}", flush=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_amp = (not args.no_amp)

    print(f"[ENV] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / "JPEGImages"
    masks_dir = dataset_root / "SegmentationClassPNG"

    print("=" * 60, flush=True)
    print("EVAL SAM2 + CLASSIFIER (MAJORITY VOTE REGION)", flush=True)
    print(f"dataset_root: {dataset_root}", flush=True)
    print(f"splits_dir  : {splits_dir}", flush=True)
    print(f"images_dir exists: {images_dir.exists()}", flush=True)
    print(f"masks_dir  exists: {masks_dir.exists()}", flush=True)
    print("=" * 60, flush=True)

    print("[SAM2] importing...", flush=True)
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    print("[SAM2] imports OK", flush=True)

    print(f"[SAM2] ckpt exists: {Path(args.sam2_ckpt).exists()} ({args.sam2_ckpt})", flush=True)
    print(f"[SAM2] config: {args.sam2_config}", flush=True)

    print("[SAM2] building model...", flush=True)
    sam2_model = build_sam2(
        config_file=args.sam2_config,
        ckpt_path=args.sam2_ckpt,
        device=device,
    )
    print("[SAM2] model built", flush=True)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )
    print("[SAM2] generator ready", flush=True)

    model = load_model(Path(args.model_ckpt), device)
    print("[Model] ready", flush=True)

    eval_ids = load_eval_ids(splits_dir, args.n_images, args.seed)
    print(f"[Data] {len(eval_ids)} images sélectionnées", flush=True)

    confmat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    results = []

    for i, sample_id in enumerate(eval_ids):
        print(f"\n[{i+1:02d}/{len(eval_ids)}] {sample_id}", flush=True)

        img_path = images_dir / f"{sample_id}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{sample_id}.JPG"
        mask_path = masks_dir / f"{sample_id}.png"

        if not img_path.exists() or not mask_path.exists():
            print(f"[WARN] manquant image/mask pour {sample_id}", flush=True)
            continue

        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        gt_rgb = np.array(Image.open(mask_path).convert("RGB"))
        gt_label = rgb_mask_to_label(gt_rgb)

        print("  [SAM2] generate...", flush=True)
        sam_masks = mask_generator.generate(image_rgb)
        print(f"  [SAM2] {len(sam_masks)} masks", flush=True)

        pred_label, kept, ignored = build_semantic_mask_vote_overlap_bestscore(
            image_rgb=image_rgb,
            sam_masks=sam_masks,
            model=model,
            device=device,
            max_masks_per_image=args.max_masks_per_image,
            region_size=args.region_size,
            use_amp=use_amp,
        )
        print(f"  [CLS] done | kept={kept} ignored={ignored}", flush=True)

        pred_t = torch.from_numpy(pred_label).long()
        gt_t = torch.from_numpy(gt_label).long()

        confmat_img = update_confusion_matrix(
            torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64),
            pred_t, gt_t, NUM_CLASSES, IGNORE_INDEX
        )
        m_img = compute_iou_from_confmat(confmat_img, IGNORE_INDEX)
        print(f"  [METRIC] mIoU={float(m_img['miou']):.4f} | pixel_acc={float(m_img['pixel_acc']):.4f}", flush=True)

        confmat = update_confusion_matrix(confmat, pred_t, gt_t, NUM_CLASSES, IGNORE_INDEX)
        results.append({
            "id": sample_id,
            "n_masks": int(len(sam_masks)),
            "kept_regions": int(kept),
            "ignored_regions": int(ignored),
            "miou": float(m_img["miou"]),
            "pixel_acc": float(m_img["pixel_acc"]),
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    m = compute_iou_from_confmat(confmat, IGNORE_INDEX)
    pixel_acc = float(m["pixel_acc"])
    miou = float(m["miou"])
    per_class_iou = [float(v) for v in m["per_class_iou"].tolist()]

    print("\n" + "=" * 60, flush=True)
    print(f"FINAL RESULTS ({len(results)} images)", flush=True)
    print(f"Pixel Accuracy: {pixel_acc:.4f}", flush=True)
    print(f"mIoU          : {miou:.4f}", flush=True)
    for c in range(1, NUM_CLASSES):
        print(f"C{c:02d} {CLASS_NAMES.get(c, f'class_{c}'):<15} IoU={per_class_iou[c]:.4f}", flush=True)
    print("=" * 60, flush=True)

    dump_json(out_dir / "results_sam_classifier.json", {
        "method": "SAM2 + region classifier (majority-vote region, overlap=best-score)",
        "n_images_requested": int(args.n_images),
        "n_images_processed": int(len(results)),
        "pixel_acc": pixel_acc,
        "miou": miou,
        "per_class_iou": per_class_iou,
        "per_image": results,
        "config": {
            "sam2_config": args.sam2_config,
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "min_mask_region_area": args.min_mask_region_area,
            "max_masks_per_image": args.max_masks_per_image,
            "region_size": args.region_size,
            "amp": bool(use_amp),
            "device": str(device),
        },
    })
    print(f"[SAVE] {out_dir / 'results_sam_classifier.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL] Exception in eval_sam.py", flush=True)
        print(str(e), flush=True)
        traceback.print_exc()
        raise
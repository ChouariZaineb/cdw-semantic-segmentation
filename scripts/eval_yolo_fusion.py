"""
scripts/eval_yolo_fusion.py

Évaluation de la fusion pondérée YOLO-seg + ResNet50+FPN.

Pipeline :
    1. ResNet50+FPN prédit des logits denses [B, 11, H, W]
    2. YOLOv8-seg prédit des masques d'instances → convertis en logits sémantiques [1, 11, H, W]
    3. Fusion : alpha × FPN_probs + (1-alpha) × YOLO_probs → argmax → masque final
    4. Métriques : mIoU, pixel accuracy, IoU par classe

Usage :
    python scripts/eval_yolo_fusion.py \
        --dataset-root  /path/to/Ground_Truths_VOC_Format \
        --splits-dir    splits/ \
        --model-ckpt    model_phase3_best_ep15.pt \
        --yolo-ckpt     runs/yolo/weights/best.pt \
        --alpha         0.12 \
        --split         test \
        --out-dir       runs/yolo_fusion \
        --device        auto
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

print("[BOOT] eval_yolo_fusion.py loaded", flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.cdw_dataset import CDWSegmentationDataset
from src.data.transforms import SegmentationBasicTransform, SegmentationTransformConfig
from src.models.build_model import build_model
from src.training.metrics import update_confusion_matrix, compute_iou_from_confmat
from scripts.utils import dump_json

# ──────────────────────────────────────────────────────────────────────────────
NUM_CLASSES    = 11
IGNORE_INDEX   = 0
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)
IMAGE_SIZE     = 768

CLASS_NAMES = {
    0: "_background_",
    1: "BIN",  2: "CB",  3: "FD",  4: "WT",  5: "HP",
    6: "SP",   7: "ST",  8: "FB",  9: "CP",  10: "PB",
}

# Palette couleur RGB cohérente avec les masques VOC
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # 0  background (noir)
    [128, 0,   0  ],   # 1  BIN
    [0,   128, 0  ],   # 2  CB
    [128, 128, 0  ],   # 3  FD
    [0,   0,   128],   # 4  WT
    [128, 0,   128],   # 5  HP
    [0,   128, 128],   # 6  SP
    [128, 128, 128],   # 7  ST
    [64,  0,   0  ],   # 8  FB
    [192, 0,   0  ],   # 9  CP
    [64,  128, 0  ],   # 10 PB
], dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def load_fpn_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    print(f"[FPN] loading: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "student")
    model = build_model(arch=arch, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval().to(device)
    print(f"[FPN] arch={arch} | val_miou={ckpt.get('val_miou', 'N/A')}", flush=True)
    return model


def predict_fpn(
    image_t: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Prédit les logits FPN pour un batch d'images prétraitées.
    image_t : [B, 3, H, W] déjà normalisé
    Retourne : [B, 11, H, W]
    """
    image_t = image_t.to(device)
    with torch.no_grad():
        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(image_t)
        else:
            logits = model(image_t)
    return logits.float()


def predict_yolo_semantic(
    image_pil: Image.Image,
    yolo_model,
    device: torch.device,
    target_size: int = IMAGE_SIZE,
    tmp_path: str = "/tmp/_yolo_eval_tmp.jpg",
) -> torch.Tensor:
    """
    Prédit un masque sémantique YOLO pour une image PIL.
    Retourne un tenseur de logits approximatifs [1, 11, target_size, target_size].

    Stratégie :
      - YOLO détecte N instances avec classes + confidences
      - Pour chaque instance : on place la confiance dans le canal de la classe YOLO
      - En cas d'overlap : la confiance la plus haute gagne (pixel-wise)
      - Le canal 0 (background) reste à 0 → softmax le forcera en background si
        aucune instance n'est détectée
    """
    # Sauvegarder image temporairement pour YOLO (prend un chemin ou un np.array)
    image_pil.save(tmp_path)
    results = yolo_model(tmp_path, imgsz=640, verbose=False)

    # Tenseur logits sémantiques vide
    yolo_semantic  = torch.zeros(NUM_CLASSES, target_size, target_size)
    confidence_map = torch.zeros(target_size, target_size)

    result = results[0]
    if result.masks is not None:
        masks   = result.masks.data.cpu()     # (N, h, w) binaire
        classes = result.boxes.cls.cpu()      # (N,) YOLO id 0-based
        confs   = result.boxes.conf.cpu()     # (N,) confiance

        for i in range(len(masks)):
            yolo_cls = int(classes[i].item())
            coco_cls = yolo_cls + 1           # YOLO 0-based → COCO/modèle 1-based
            conf     = confs[i].item()

            # Redimensionner masque binaire → target_size
            mask = masks[i].unsqueeze(0).unsqueeze(0).float()
            mask = F.interpolate(
                mask, size=(target_size, target_size),
                mode="bilinear", align_corners=False
            ).squeeze()

            # Mise à jour pixel-wise : confiance la plus haute gagne
            update = (mask > 0.5) & (conf > confidence_map)
            confidence_map[update] = conf
            yolo_semantic[coco_cls][update] = conf

    return yolo_semantic.unsqueeze(0).to(device)  # [1, 11, H, W]


def fuse_predictions(
    fpn_logits: torch.Tensor,
    yolo_logits: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Fusion pondérée :
      alpha       → poids FPN  (modèle sémantique dense, plus fiable globalement)
      (1 - alpha) → poids YOLO (segmentation d'instances, plus précis sur objets)

    Retourne les prédictions finales [B, H, W] (argmax).
    """
    fpn_probs  = torch.softmax(fpn_logits,  dim=1)
    yolo_probs = torch.softmax(yolo_logits, dim=1)
    fused      = alpha * fpn_probs + (1.0 - alpha) * yolo_probs
    return torch.argmax(fused, dim=1)


def label_to_color(label: np.ndarray) -> np.ndarray:
    """[H, W] int → [H, W, 3] RGB"""
    return CLASS_COLORS[np.clip(label, 0, NUM_CLASSES - 1)]


def apply_background_mask(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Force les pixels background du GT à 0 dans la prédiction (visualisation seulement)."""
    result = pred.copy()
    result[gt == IGNORE_INDEX] = IGNORE_INDEX
    return result


def save_comparison_image(
    image_pil: Image.Image,
    gt_label: np.ndarray,
    yolo_mask: np.ndarray,
    fpn_mask: np.ndarray,
    fused_mask: np.ndarray,
    sample_id: str,
    miou_yolo: float,
    miou_fpn: float,
    miou_fused: float,
    out_dir: Path,
) -> None:
    """
    Sauvegarde une image de comparaison 5 panneaux :
      [Image originale | GT | YOLO | FPN | Fusion]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    gt_color    = label_to_color(gt_label)
    yolo_color  = label_to_color(apply_background_mask(yolo_mask,  gt_label))
    fpn_color   = label_to_color(apply_background_mask(fpn_mask,   gt_label))
    fused_color = label_to_color(apply_background_mask(fused_mask, gt_label))

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.patch.set_facecolor("#1a1a1a")

    cols = [
        (np.array(image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))), "Image originale",          None),
        (gt_color,    "Ground Truth",                                                       None),
        (yolo_color,  f"YOLO  (mIoU={miou_yolo:.3f})",                                     None),
        (fpn_color,   f"FPN   (mIoU={miou_fpn:.3f})",                                      None),
        (fused_color, f"Fusion α={args_alpha:.2f}  (mIoU={miou_fused:.3f})",               None),
    ]

    for ax, (img, title, _) in zip(axes, cols):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.axis("off")

    # Légende — classes présentes dans GT ou prédictions
    classes_present = (
        set(np.unique(gt_label).tolist())
        | set(np.unique(yolo_mask).tolist())
        | set(np.unique(fpn_mask).tolist())
        | set(np.unique(fused_mask).tolist())
    )
    classes_present.discard(IGNORE_INDEX)

    patches = [
        mpatches.Patch(
            color=np.array(CLASS_COLORS[c]) / 255.0,
            label=f"C{c} {CLASS_NAMES.get(c, f'class_{c}')}"
        )
        for c in sorted(classes_present)
    ]
    if patches:
        fig.legend(
            handles=patches, loc="lower center",
            ncol=min(len(patches), 5), fontsize=8,
            framealpha=0.3, facecolor="#333333", labelcolor="white",
            bbox_to_anchor=(0.5, -0.04),
        )

    plt.suptitle(sample_id, color="white", fontsize=9, y=1.01)
    plt.tight_layout()

    out_path = out_dir / f"{sample_id}_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [VIZ] saved → {out_path}", flush=True)


# variable globale pour save_comparison_image (alpha)
args_alpha: float = 0.12


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global args_alpha

    os.environ["PYTHONUNBUFFERED"] = "1"
    print("[BOOT] entering main()", flush=True)

    parser = argparse.ArgumentParser(
        description="Évaluation fusion YOLO-seg + ResNet50+FPN"
    )
    parser.add_argument("--dataset-root", required=True,
                        help="Racine du dataset VOC (JPEGImages/ + SegmentationClassPNG/)")
    parser.add_argument("--splits-dir",   required=True,
                        help="Dossier contenant train.txt / val.txt / test.txt")
    parser.add_argument("--model-ckpt",   required=True,
                        help="Checkpoint ResNet50+FPN (.pt)")
    parser.add_argument("--yolo-ckpt",    required=True,
                        help="Checkpoint YOLOv8-seg (.pt)")
    parser.add_argument("--alpha",        type=float, default=0.12,
                        help="Poids du FPN dans la fusion (défaut=0.12 → FPN 12%%, YOLO 88%%)")
    parser.add_argument("--split",        default="test",
                        choices=["train", "val", "test"],
                        help="Split à évaluer (défaut=test)")
    parser.add_argument("--batch-size",   type=int, default=2)
    parser.add_argument("--num-workers",  type=int, default=2)
    parser.add_argument("--image-size",   type=int, default=IMAGE_SIZE)
    parser.add_argument("--out-dir",      default="runs/yolo_fusion")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--no-amp",       action="store_true")
    parser.add_argument("--no-viz",       action="store_true",
                        help="Désactiver la sauvegarde des images de comparaison")
    args = parser.parse_args()

    args_alpha = args.alpha

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_amp = not args.no_amp

    print(f"[ENV] device={device} amp={use_amp}", flush=True)
    print(f"[CFG] alpha={args.alpha} (FPN={args.alpha:.0%} / YOLO={1-args.alpha:.0%})", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Charger modèles ──
    fpn_model  = load_fpn_model(Path(args.model_ckpt), device)

    print("[YOLO] loading...", flush=True)
    from ultralytics import YOLO
    yolo_model = YOLO(args.yolo_ckpt)
    print("[YOLO] ready", flush=True)

    # ── Dataset ──
    splits_dir = Path(args.splits_dir)
    split_file = splits_dir / f"{args.split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file non trouvé : {split_file}")

    tfm = SegmentationBasicTransform(
        SegmentationTransformConfig(
            size=(args.image_size, args.image_size),
            keep_aspect_ratio=True,
        )
    )

    dataset = CDWSegmentationDataset(
        dataset_root=Path(args.dataset_root),
        split_file=split_file,
        transform=tfm,
        oversample_rare=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"[Data] {len(dataset)} images ({args.split})", flush=True)

    # ── Matrices de confusion ──
    confmat_fpn   = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    confmat_yolo  = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    confmat_fused = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    results_per_image: list[dict] = []

    print("=" * 60, flush=True)
    print(f"EVAL YOLO-FUSION — split={args.split}", flush=True)
    print("=" * 60, flush=True)

    fpn_model.eval()
    images_dir = Path(args.dataset_root) / "JPEGImages"

    for images_t, masks_t, sample_ids in tqdm(dataloader, desc=f"Eval {args.split}"):
        # ── FPN (batch) ──
        fpn_logits = predict_fpn(images_t, fpn_model, device, use_amp)   # [B, 11, H, W]
        fpn_preds  = torch.argmax(fpn_logits, dim=1).cpu()               # [B, H, W]
        confmat_fpn = update_confusion_matrix(
            confmat_fpn, fpn_preds, masks_t, NUM_CLASSES, IGNORE_INDEX
        )

        fpn_probs = torch.softmax(fpn_logits, dim=1)

        # ── YOLO (image par image) ──
        for b, sample_id in enumerate(sample_ids):
            img_path = images_dir / f"{sample_id}.jpg"
            if not img_path.exists():
                img_path = images_dir / f"{sample_id}.JPG"

            img_pil = Image.open(img_path).convert("RGB")

            yolo_logits = predict_yolo_semantic(
                img_pil, yolo_model, device, target_size=args.image_size
            )
            yolo_probs = torch.softmax(yolo_logits, dim=1)

            # YOLO seul
            yolo_pred = torch.argmax(yolo_probs, dim=1).cpu()
            confmat_yolo = update_confusion_matrix(
                confmat_yolo, yolo_pred, masks_t[b].unsqueeze(0), NUM_CLASSES, IGNORE_INDEX
            )

            # Fusion
            fpn_prob_single = fpn_probs[b].unsqueeze(0)
            fused_pred = fuse_predictions(fpn_prob_single, yolo_logits, args.alpha).cpu()
            confmat_fused = update_confusion_matrix(
                confmat_fused, fused_pred, masks_t[b].unsqueeze(0), NUM_CLASSES, IGNORE_INDEX
            )

            # ── Métriques par image ──
            gt_np    = masks_t[b].numpy()
            fpn_np   = fpn_preds[b].numpy()
            yolo_np  = yolo_pred.squeeze(0).numpy()
            fused_np = fused_pred.squeeze(0).numpy()

            def img_miou(pred, gt):
                cf = update_confusion_matrix(
                    torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64),
                    torch.from_numpy(pred).unsqueeze(0),
                    torch.from_numpy(gt).unsqueeze(0),
                    NUM_CLASSES, IGNORE_INDEX,
                )
                return float(compute_iou_from_confmat(cf, IGNORE_INDEX)["miou"])

            m_fpn   = img_miou(fpn_np,   gt_np)
            m_yolo  = img_miou(yolo_np,  gt_np)
            m_fused = img_miou(fused_np, gt_np)

            print(
                f"  [{sample_id}] "
                f"FPN={m_fpn:.4f} | YOLO={m_yolo:.4f} | Fusion={m_fused:.4f}",
                flush=True,
            )

            results_per_image.append({
                "id":         sample_id,
                "miou_fpn":   m_fpn,
                "miou_yolo":  m_yolo,
                "miou_fused": m_fused,
            })

            # ── Sauvegarde image de comparaison ──
            if not args.no_viz:
                save_comparison_image(
                    image_pil=img_pil,
                    gt_label=gt_np,
                    yolo_mask=yolo_np,
                    fpn_mask=fpn_np,
                    fused_mask=fused_np,
                    sample_id=sample_id,
                    miou_yolo=m_yolo,
                    miou_fpn=m_fpn,
                    miou_fused=m_fused,
                    out_dir=out_dir,
                )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Résultats globaux ──
    m_fpn   = compute_iou_from_confmat(confmat_fpn,   IGNORE_INDEX)
    m_yolo  = compute_iou_from_confmat(confmat_yolo,  IGNORE_INDEX)
    m_fused = compute_iou_from_confmat(confmat_fused, IGNORE_INDEX)

    print("\n" + "=" * 60, flush=True)
    print(f"RÉSULTATS FINAUX — split={args.split}", flush=True)
    print(f"{'Modèle':<25} {'mIoU':>8}  {'Pixel Acc':>10}", flush=True)
    print("-" * 60, flush=True)
    print(f"{'FPN seul':<25} {float(m_fpn['miou']):.4f}    {float(m_fpn['pixel_acc']):.4f}", flush=True)
    print(f"{'YOLO seul':<25} {float(m_yolo['miou']):.4f}    {float(m_yolo['pixel_acc']):.4f}", flush=True)
    print(f"{f'Fusion α={args.alpha}':<25} {float(m_fused['miou']):.4f}    {float(m_fused['pixel_acc']):.4f}", flush=True)
    print("=" * 60, flush=True)
    print(f"📈 Gain Fusion vs FPN : +{float(m_fused['miou']) - float(m_fpn['miou']):.4f}", flush=True)

    print(f"\n{'Classe':<12} {'FPN':>8}  {'YOLO':>8}  {'Fusion':>8}  {'Meilleur':>10}", flush=True)
    print("-" * 60, flush=True)

    per_class_results: list[dict] = []
    for c in range(1, NUM_CLASSES):
        iou_fpn   = float(m_fpn["per_class_iou"][c])
        iou_yolo  = float(m_yolo["per_class_iou"][c])
        iou_fused = float(m_fused["per_class_iou"][c])
        best      = max(iou_fpn, iou_yolo, iou_fused)
        winner    = "Fusion" if iou_fused == best else ("FPN" if iou_fpn == best else "YOLO")
        print(
            f"  {CLASS_NAMES[c]:<10} {iou_fpn:>8.4f}  {iou_yolo:>8.4f}"
            f"  {iou_fused:>8.4f}  {winner:>10}",
            flush=True,
        )
        per_class_results.append({
            "class": CLASS_NAMES[c],
            "iou_fpn":   iou_fpn,
            "iou_yolo":  iou_yolo,
            "iou_fused": iou_fused,
            "winner":    winner,
        })

    print("=" * 60, flush=True)

    # ── Export JSON ──
    dump_json(out_dir / "results_yolo_fusion.json", {
        "method":     "YOLO-seg + FPN fusion (weighted probs)",
        "split":      args.split,
        "alpha":      args.alpha,
        "global": {
            "fpn":   {"miou": float(m_fpn["miou"]),   "pixel_acc": float(m_fpn["pixel_acc"])},
            "yolo":  {"miou": float(m_yolo["miou"]),  "pixel_acc": float(m_yolo["pixel_acc"])},
            "fused": {"miou": float(m_fused["miou"]), "pixel_acc": float(m_fused["pixel_acc"])},
        },
        "per_class":    per_class_results,
        "per_image":    results_per_image,
        "config": {
            "model_ckpt":   args.model_ckpt,
            "yolo_ckpt":    args.yolo_ckpt,
            "image_size":   args.image_size,
            "batch_size":   args.batch_size,
            "device":       str(device),
            "amp":          use_amp,
        },
    })
    print(f"[SAVE] {out_dir / 'results_yolo_fusion.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL] Exception in eval_yolo_fusion.py", flush=True)
        print(str(e), flush=True)
        traceback.print_exc()
        raise
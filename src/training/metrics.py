from __future__ import annotations
"""Confusion-matrix based metrics for segmentation evaluation."""

import torch


@torch.no_grad()
def update_confusion_matrix(
    confmat: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Met à jour la matrice de confusion.

    Args:
        confmat:
            Tensor [C, C], où lignes = vérité terrain, colonnes = prédiction.
        preds:
            Tensor [B, H, W] des prédictions (labels entiers).
        targets:
            Tensor [B, H, W] des labels réels.
        num_classes:
            Nombre total de labels, y compris ignore_index si présent dans l'intervalle.
            Exemple: 11 pour labels 0..10.
        ignore_index:
            Label ignoré dans les cibles. Ici: 0.

    Returns:
        confmat mise à jour.
    """
    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    in_range = (
        (targets >= 0) & (targets < num_classes) &
        (preds >= 0) & (preds < num_classes)
    )
    preds = preds[in_range]
    targets = targets[in_range]

    idx = targets * num_classes + preds
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    confmat += binc.reshape(num_classes, num_classes)
    return confmat


@torch.no_grad()
def compute_iou_from_confmat(
    confmat: torch.Tensor,
    ignore_index: int = 0,
) -> dict:
    """
    Calcule les métriques de segmentation à partir d'une matrice de confusion.

    Important:
    - ignore_index est exclu du calcul du mIoU
    - pixel accuracy est calculée uniquement sur les pixels valides
      (ce qui est déjà garanti si update_confusion_matrix a filtré ignore_index)

    Returns:
        dict avec:
        - per_class_iou: Tensor [C]
        - miou: scalaire
        - pixel_acc: scalaire
        - valid_pixel_count: nombre de pixels réellement évalués
    """
    confmat = confmat.to(torch.float64)

    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp

    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))

    valid_pixel_count = confmat.sum()
    correct = tp.sum()
    pixel_acc = (
        correct / valid_pixel_count
        if valid_pixel_count > 0
        else torch.tensor(0.0, dtype=torch.float64)
    )

    valid_classes = [c for c in range(confmat.shape[0]) if c != ignore_index]
    if len(valid_classes) > 0:
        miou = iou[valid_classes].mean()
    else:
        miou = torch.tensor(0.0, dtype=torch.float64)

    return {
        "per_class_iou": iou,
        "miou": miou,
        "pixel_acc": pixel_acc,
        "valid_pixel_count": valid_pixel_count,
    }
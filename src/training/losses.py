from __future__ import annotations
"""Loss functions used for semantic segmentation training."""

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_weighted_ce_loss(
    num_classes: int,
    class_weights: Optional[Iterable[float]] = None,
    ignore_index: int = 0,
) -> nn.Module:
    """
    Weighted Cross-Entropy for semantic segmentation.

    Args:
        num_classes:
            Nombre total de classes, y compris le label ignoré.
            Exemple: 11 pour les labels 0..10.

        class_weights:
            Poids par classe de longueur num_classes.
            Si None: loss non pondérée.
            Si ignore_index=0, le poids de la classe 0 n'a pas d'effet pratique
            car cette classe est ignorée dans la loss.

        ignore_index:
            Label ignoré dans le masque cible.
            Dans ce projet: 0 = pixels non annotés / padding / ignore label.

    Returns:
        nn.CrossEntropyLoss configurée pour la segmentation.
    """
    weight_tensor = None

    if class_weights is not None:
        weights = list(class_weights)

        if len(weights) != num_classes:
            raise ValueError(
                f"class_weights must have length {num_classes}, got {len(weights)}"
            )

        weight_tensor = torch.tensor(weights, dtype=torch.float32)

    loss = nn.CrossEntropyLoss(
        weight=weight_tensor,
        ignore_index=ignore_index,
    )

    return loss


class FocalLoss(nn.Module):
    """
    Focal Loss pour segmentation sémantique multi-classe.

    Formule :
        FL(p, y) = -α_y × (1 - p_y)^γ × log(p_y)

    Par rapport à la Cross-Entropy pondérée :
        - α (class_weights) → corrige le déséquilibre de FRÉQUENCE entre classes
        - γ (gamma)         → corrige le déséquilibre de DIFFICULTÉ entre pixels

    Les deux mécanismes sont complémentaires — α seul ne suffit pas
    quand certaines classes sont à la fois rares ET visuellement difficiles
    (ex: hard_plastic C5, fabric C8, plasterboard C10).

    Args:
        num_classes:
            Nombre total de classes (ex: 11 pour labels 0..10).

        class_weights:
            Poids par classe (tes poids MFB existants).
            Même format que make_weighted_ce_loss.
            Si None : pas de pondération par classe (alpha=1 pour tout).

        gamma:
            Paramètre focal. Contrôle la pénalisation des exemples faciles.
            - γ=0 → équivalent à la Cross-Entropy classique
            - γ=1 → réduction modérée des exemples faciles
            - γ=2 → valeur recommandée (Lin et al. 2017, papier original)
            - γ=5 → réduction très agressive (risque d'instabilité)
            Recommandation pour ton cas : commencer à γ=2.

        ignore_index:
            Label ignoré (0 = pixels non annotés dans ton projet).
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[Iterable[float]] = None,
        gamma: float = 2.0,
        ignore_index: int = 0,
    ):
        super().__init__()

        self.num_classes   = num_classes
        self.gamma         = gamma
        self.ignore_index  = ignore_index

        # Enregistrer les poids comme buffer (déplacé automatiquement
        # sur GPU avec .to(device) ou .cuda())
        if class_weights is not None:
            weights = list(class_weights)
            if len(weights) != num_classes:
                raise ValueError(
                    f"class_weights must have length {num_classes}, "
                    f"got {len(weights)}"
                )
            self.register_buffer(
                "alpha",
                torch.tensor(weights, dtype=torch.float32)
            )
        else:
            self.register_buffer("alpha", None)

    def forward(
        self,
        inputs: torch.Tensor,   # [B, C, H, W] logits bruts
        targets: torch.Tensor,  # [B, H, W]    labels entiers
    ) -> torch.Tensor:
        """
        Args:
            inputs:  logits bruts [B, C, H, W] — pas de softmax avant
            targets: masque de labels [B, H, W] avec valeurs 0..num_classes-1

        Returns:
            Scalar loss (moyenne sur les pixels non ignorés)
        """

        # ── Étape 1 : Cross-Entropy pixel par pixel (sans réduction) ─────────
        # reduction='none' → on garde la loss par pixel [B, H, W]
        # pour pouvoir appliquer le facteur focal ensuite
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,          # tes poids MFB (α)
            ignore_index=self.ignore_index,
            reduction='none',           # IMPORTANT : pas de moyenne ici
        )
        # ce_loss : [B, H, W]

        # ── Étape 2 : Calculer p_y (probabilité de la vraie classe) ──────────
        # exp(-CE) = exp(log(p_y)) = p_y
        # Plus stable numériquement que softmax + gather
        p_y = torch.exp(-ce_loss)
        # p_y : [B, H, W], valeurs dans [0, 1]

        # ── Étape 3 : Appliquer le facteur focal (1 - p_y)^γ ─────────────────
        # Quand p_y ≈ 1 (pixel facile)    → facteur ≈ 0 → loss réduite
        # Quand p_y ≈ 0 (pixel difficile) → facteur ≈ 1 → loss intacte
        focal_factor = (1.0 - p_y) ** self.gamma
        focal_loss   = focal_factor * ce_loss
        # focal_loss : [B, H, W]

        # ── Étape 4 : Moyenne sur les pixels NON ignorés uniquement ──────────
        # Les pixels ignore_index ont ce_loss=0 (PyTorch les met à 0)
        # → on divise par le nombre de pixels valides, pas le total
        valid_mask   = targets != self.ignore_index  # [B, H, W] booléen
        n_valid      = valid_mask.sum().clamp(min=1)  # éviter division par 0
        loss         = focal_loss[valid_mask].sum() / n_valid

        return loss

    def __repr__(self) -> str:
        return (
            f"FocalLoss("
            f"num_classes={self.num_classes}, "
            f"gamma={self.gamma}, "
            f"ignore_index={self.ignore_index}, "
            f"weighted={'yes' if self.alpha is not None else 'no'})"
        )


def make_focal_loss(
    num_classes: int,
    class_weights: Optional[Iterable[float]] = None,
    gamma: float = 2.0,
    ignore_index: int = 0,
) -> FocalLoss:
    """
    Factory function pour la Focal Loss — même interface que make_weighted_ce_loss.

    Args:
        num_classes:   Nombre total de classes (ex: 11).
        class_weights: Poids MFB par classe (même que make_weighted_ce_loss).
        gamma:         Paramètre focal (défaut: 2.0, recommandé par Lin et al.).
        ignore_index:  Label ignoré (défaut: 0).

    Returns:
        FocalLoss configurée et prête à l'emploi.

    Exemple d'utilisation dans ton script d'entraînement :

        # Remplacer :
        loss_fn = make_weighted_ce_loss(
            num_classes=11,
            class_weights=weights,
            ignore_index=0,
        )

        # Par :
        loss_fn = make_focal_loss(
            num_classes=11,
            class_weights=weights,
            gamma=2.0,
            ignore_index=0,
        )
    """
    return FocalLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        gamma=gamma,
        ignore_index=ignore_index,
    )
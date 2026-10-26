# PROJECT FILES DESCRIPTION

## Vue d'ensemble

Ce projet met en place une pipeline de segmentation sémantique pour des images CDW (construction / déchets / matériaux), avec 11 classes au total et la classe `0` utilisée comme `ignore_index`.

La méthode principale repose sur :
- un dataset de style VOC (`JPEGImages/` + `SegmentationClassPNG/`)
- un modèle de segmentation basé sur un backbone ResNet avec FPN
- un entraînement en 3 phases avec dégel progressif du backbone
- une loss pondérée pour gérer le déséquilibre de classes
- une évaluation classique par IoU / mIoU
- deux variantes d'évaluation SAM2: une version région seule et une version avec fallback dense
- une branche expérimentale YOLO-seg avec conversion COCO et fusion pondérée avec le modèle FPN

## Méthode utilisée

### Entraînement principal

Le script d'entraînement suit trois phases :
- Phase 1: backbone gelé, entraînement de la tête
- Phase 2: dégel des blocs profonds du backbone (`layer3` et `layer4`)
- Phase 3: dégel complet, avec early stopping basé sur la mIoU de validation

L'entraînement utilise :
- des transformations adaptées à la segmentation
- un sur-échantillonnage des classes rares
- une Cross-Entropy pondérée par fréquence médiane
- deux taux d'apprentissage séparés pour backbone et tête
- des métriques de validation basées sur la matrice de confusion

### Modèle

Le modèle principal est un segmenter FPN :
- backbone ResNet50 / ResNet101 / ResNet152 selon l'alias choisi
- features pyramidales fusionnées
- sortie dense par convolution `1x1`
- rééchantillonnage final à la taille d'entrée

### Évaluation classique

Le script `scripts/eval.py` charge un checkpoint, reconstruit le modèle, parcourt un split donné et exporte :
- pixel accuracy
- mIoU
- IoU par classe
- matrice de confusion

### Évaluation SAM2

Le dépôt contient deux scripts proches mais distincts :

- `scripts/eval_sam.py` : évaluation SAM2 + classifieur de régions.
  - SAM2 génère des masques de régions
  - chaque région est classée par le modèle de segmentation
  - les overlaps sont résolus par le meilleur score de majorité
  - la sortie peut rester partielle si SAM2 ne couvre pas tous les pixels
- `scripts/eval_sam_m2.py` : variante SAM2 + classifieur de régions + fallback dense.
  - reprend la logique de `eval_sam.py`
  - ajoute une passe dense sur l'image entière
  - comble les pixels non couverts par SAM2 avec la prédiction dense
  - sauvegarde aussi des images de comparaison GT / prédiction finale

### Méthode YOLO

Le dépôt contient aussi une branche expérimentale basée sur YOLO segmentation :

- `scripts/convert_coco_to_yolo.py` : conversion des annotations COCO vers le format YOLO-seg.
  - lit `annotations.json` et `JPEGImages/`
  - ignore la classe `0` considérée comme background
  - convertit les annotations en polygones normalisés YOLO
  - répartit les images en `train/` et `val/` avec un split 80/20
  - génère automatiquement un `data.yaml` compatible YOLO

- `scripts/eval_yolo_fusion.py` : évaluation d’une fusion pondérée YOLO-seg + ResNet50+FPN.
  - charge un checkpoint FPN supervisé et un checkpoint YOLOv8-seg
  - transforme les masques d'instances YOLO en logits sémantiques
  - fusionne les probabilités avec un poids `alpha`
  - calcule mIoU, pixel accuracy et IoU par classe
  - sauvegarde aussi des visualisations de comparaison GT / YOLO / FPN / fusion

## Arborescence détaillée

### Racine du projet

- `requirements.txt` : liste minimale des dépendances Python.
- `model_phase3_best_ep15.pt` : checkpoint entraîné, vraisemblablement le meilleur modèle de la phase 3 à l'époque `ep15`.
- `PROJECT_FILES_DESCRIPTION.md` : ce document.

### `configs/`

Contient les fichiers de configuration YAML.

- `configs/train/multiphase.yaml` : configuration par défaut de l'entraînement multiphase.
  - définit le nom d'expérience
  - fixe `student` comme architecture de base
  - configure taille d'image, batch size, patience, epochs et learning rates
  - laisse `dataset_root` vide pour être fourni à l'exécution
- `configs/inference/default.yaml` : configuration par défaut pour l'inférence classique.
  - fixe classes, ignore index, taille d'entrée et batch size
  - utilise `device: auto`
  - écrit les résultats dans `results/`

### `scripts/`

Scripts exécutables de haut niveau.

- `scripts/train.py` : point d'entrée principal d'entraînement.
  - charge la config YAML et les overrides CLI
  - construit dataset, transforms, modèle, loss, optimizer
  - lance les 3 phases d'entraînement
  - sauvegarde checkpoints, métriques JSONL et résumé final
- `scripts/eval.py` : évaluation d'un checkpoint sur un split de validation/test.
  - reconstruit le dataset et le modèle
  - calcule pixel accuracy, mIoU, IoU par classe
  - exporte un JSON de rapport
- `scripts/eval_sam.py` : évaluation SAM2 région seule.
  - génère des masques SAM2
  - classe chaque région par vote majoritaire via le modèle
  - résout les overlaps par meilleur score
  - exporte un JSON de résultats partiels
- `scripts/eval_sam_m2.py` : évaluation SAM2 + classifieur de régions + fallback dense.
  - génère des masques SAM2
  - attribue une classe à chaque région par vote majoritaire
  - ajoute une passe dense de secours sur les pixels non couverts
  - produit des visualisations comparatives
  - exporte un JSON de résultats détaillés
- `scripts/convert_coco_to_yolo.py` : conversion COCO → YOLO segmentation.
  - transforme les annotations en polygones YOLO normalisés
  - crée automatiquement les dossiers `images/` et `labels/`
  - écrit un `data.yaml` prêt pour l'entraînement YOLO
- `scripts/eval_yolo_fusion.py` : évaluation de la fusion YOLO-seg + FPN.
  - charge un modèle YOLOv8-seg et un modèle FPN
  - convertit les prédictions YOLO en carte sémantique
  - fusionne les deux sources de prédiction par moyenne pondérée
  - exporte métriques globales, IoU par classe et visualisations
- `scripts/utils.py` : utilitaires transverses.
  - chargement YAML
  - fusion profonde de dictionnaires
  - seed global
  - création de dossier d'expérience
  - export JSON / YAML / JSONL

### `src/`

Code source Python réutilisable.

- `src/__init__.py` : marqueur de package.

#### `src/data/`

Code lié aux données et aux prétraitements.

- `src/data/__init__.py` : marqueur de package.
- `src/data/cdw_dataset.py` : dataset principal.
  - lit les IDs depuis un fichier de split texte
  - charge les images depuis `JPEGImages/`
  - charge les masques depuis `SegmentationClassPNG/`
  - vérifie que les masques sont 2D
  - supporte un sur-échantillonnage des classes rares
- `src/data/transforms.py` : pipelines de transformations.
  - `SegmentationBasicTransform` : resize, padding, normalisation ImageNet, conversion tensor
  - `SegmentationAugmentedTransform` : augmentations géométriques + photométriques pour l'entraînement
  - `SegmentationTransformConfig` : configuration commune des tailles et paddings

#### `src/models/`

Définition et construction des architectures.

- `src/models/__init__.py` : marqueur de package.
- `src/models/build_model.py` : factory de modèles.
  - mappe les alias d'architecture vers un backbone ResNet précis
  - construit un `FPNSegmenter`
- `src/models/fpn_segmenter.py` : modèle de segmentation FPN.
  - backbone ResNet avec FPN via `torchvision`
  - fusion des niveaux pyramidaux
  - tête de classification dense

#### `src/training/`

Utilitaires d'entraînement, métriques et pondération.

- `src/training/__init__.py` : marqueur de package.
- `src/training/losses.py` : pertes.
  - `make_weighted_ce_loss` : Cross-Entropy pondérée
  - `FocalLoss` et `make_focal_loss` : présents mais non utilisés dans la config actuelle
- `src/training/metrics.py` : métriques à base de matrice de confusion.
  - mise à jour de la confusion matrix
  - calcul IoU, mIoU et pixel accuracy
- `src/training/freeze.py` : gel / dégel du backbone.
  - gel total
  - dégel partiel des blocs C4-C5
  - dégel complet
- `src/training/class_weights.py` : calcul des poids de classes.
  - implémente la median frequency balancing
  - limite optionnelle des poids extrêmes
- `src/training/optim.py` : construction de l'optimiseur.
  - `AdamW`
  - groupes de paramètres séparés pour backbone et tête
- `src/training/pixel_stats.py` : lecture des statistiques de pixels.
  - parse `pixel_stats.txt`
  - retourne les comptages globaux et par classe

### `splits/`

Fichiers texte listant les identifiants d'images par partition.

- `splits/train.txt` : identifiants du split d'entraînement.
- `splits/val.txt` : identifiants du split de validation.
- `splits/test.txt` : identifiants du split de test.
- `splits/pixel_stats.txt` : statistiques globales de pixels, utilisées pour le calcul des poids de classes.

## Fichiers et dossiers attendus par la méthode

Le code suppose un dataset racine de type VOC contenant au minimum :
- `JPEGImages/` : images RGB
- `SegmentationClassPNG/` : masques PNG de segmentation

Le script d'entraînement utilise aussi :
- `train.txt`
- `val.txt`
- éventuellement `check.txt` si présent
- `pixel_stats.txt`

Pour la branche YOLO, la conversion attend :
- `annotations.json`
- `JPEGImages/`
- un dossier de sortie pour générer `images/`, `labels/` et `data.yaml`

## Sorties générées

Les scripts produisent généralement :
- des checkpoints `.pt`
- `metrics.jsonl`
- `summary.json`
- `config_snapshot.yaml`
- des rapports JSON d'évaluation
- des images de comparaison pour l'évaluation SAM2
- des fichiers YOLO de conversion et d'évaluation
- des visualisations comparatives YOLO / FPN / fusion

## Résumé technique court

En pratique, ce dépôt implémente une segmentation sémantique supervisée classique avec backbone ResNet + FPN, renforcée par un entraînement en plusieurs phases, une gestion du déséquilibre de classes, et un mode d'évaluation SAM2 expérimental pour comparer une approche par régions à la prédiction dense directe.
Il ajoute aussi une branche YOLO-seg pour la conversion des annotations et une évaluation hybride par fusion pondérée avec le modèle FPN.
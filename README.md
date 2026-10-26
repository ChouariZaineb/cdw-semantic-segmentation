# CDW Semantic Segmentation

Pipeline de segmentation sémantique pour des images CDW (construction, déchets et matériaux). Le projet entraîne un modèle ResNet + FPN sur 11 classes, avec la classe `0` réservée à l'`ignore_index`.

## Fonctionnalités

- Entraînement en trois phases avec dégel progressif du backbone.
- Pertes pondérées pour limiter l'effet du déséquilibre entre classes.
- Évaluation par pixel accuracy, IoU par classe et mIoU.
- Évaluation expérimentale avec SAM2, avec ou sans fallback dense.
- Conversion COCO vers YOLO-seg et fusion expérimentale YOLO + FPN.

## Structure du dépôt

```text
cdw-semantic-segmentation/
├── configs/                 # Configurations YAML d'entraînement et d'inférence
├── scripts/                 # Points d'entrée train/eval/conversion
├── src/                     # Dataset, modèle FPN et utilitaires d'entraînement
├── splits/                  # Fichiers train/val/test et statistiques de pixels
├── requirements.txt
└── model_phase3_best_ep15.pt # Checkpoint ResNet50 + FPN fourni
```

Le fichier `model_phase3_best_ep15.pt` est un checkpoint obtenu avec une architecture **ResNet50 + FPN**.


## Installation

Python 3.10 ou plus récent est recommandé. Depuis ce dossier :

```bash
python -m venv .venv
```

Activation sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Activation sous Linux/macOS :

```bash
source .venv/bin/activate
```

Installation des dépendances :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pour utiliser un GPU, installez une version de PyTorch adaptée à votre version de CUDA depuis le site officiel de PyTorch avant d'installer le reste des dépendances.

## Format du dataset

Le dataset de référence est disponible ici : [A benchmark dataset for class-wise segmentation of construction and demolition waste in cluttered environments - Springer Nature - Figshare](https://springernature.figshare.com/articles/dataset/A_benchmark_dataset_for_class-wise_segmentation_of_construction_and_demolition_waste_in_cluttered_environments/28573229).

Le dataset doit suivre une structure de type VOC :

```text
<dataset-root>/
├── JPEGImages/
│   ├── image_001.jpg
│   └── ...
└── SegmentationClassPNG/
    ├── image_001.png
    └── ...
```

Les masques doivent être des images PNG 2D dont les valeurs correspondent aux identifiants de classes. Les identifiants d'images utilisés par les splits doivent correspondre aux noms de fichiers sans extension.

Les fichiers de split se trouvent par défaut dans `splits/` :

- `train.txt` : images d'entraînement ;
- `val.txt` : images de validation ;
- `test.txt` : images de test ;
- `pixel_stats.txt` : statistiques utilisées pour calculer les poids de classes.

## Augmentation des données

L'augmentation est intégrée au pipeline d'entraînement dans `src/data/transforms.py`, via la classe `SegmentationAugmentedTransform`. Il ne s'agit pas d'un script séparé.

Les augmentations appliquées de manière aléatoire comprennent notamment :

- retournements horizontal et vertical ;
- rotations à 90 degrés ;
- translations, changements d'échelle et rotations ;
- recadrage aléatoire ;
- variations de luminosité, contraste, saturation et teinte ;
- correction gamma, flou, bruit et brouillard.

Les transformations géométriques sont appliquées simultanément à l'image et au masque. Les transformations photométriques sont appliquées uniquement à l'image. En validation et en évaluation, le pipeline standard sans augmentation est utilisé.

## Entraînement

La configuration par défaut est `configs/train/multiphase.yaml`. Elle utilise une entrée de `768 x 768`, un batch size de `2` et trois phases d'entraînement :

1. tête seule avec backbone gelé ;
2. dégel des blocs profonds du backbone ;
3. dégel complet avec early stopping sur la mIoU de validation.

Exemple :

```bash
python scripts/train.py \
  --config configs/train/multiphase.yaml \
  --dataset-root /chemin/vers/dataset \
  --splits-dir splits \
  --out-dir runs
```

Sous PowerShell, utilisez le caractère d'accent grave `` ` `` pour continuer une commande sur plusieurs lignes, ou écrivez-la sur une seule ligne :

```powershell
python scripts/train.py --config configs/train/multiphase.yaml --dataset-root "C:\data\cdw" --splits-dir splits --out-dir runs
```

Paramètres utiles :

```bash
python scripts/train.py --help
```

Les expériences génèrent généralement un checkpoint, `metrics.jsonl`, `summary.json` et une copie de la configuration utilisée.

## Évaluation classique

Évaluer le checkpoint fourni sur le split de test :

```bash
python scripts/eval.py \
  --ckpt model_phase3_best_ep15.pt \
  --dataset-root /chemin/vers/dataset \
  --split-file splits/test.txt \
  --out-json results/test_results.json \
  --device auto
```

Pour afficher toutes les options :

```bash
python scripts/eval.py --help
```

Le rapport JSON contient la pixel accuracy, la mIoU, l'IoU de chaque classe et la matrice de confusion.

## Évaluations expérimentales

### SAM2

Les scripts `scripts/eval_sam.py` et `scripts/eval_sam_m2.py` nécessitent une installation séparée de SAM2, sa configuration et un checkpoint SAM2. Ils prennent également un checkpoint du modèle FPN.

```bash
python scripts/eval_sam.py \
  --dataset-root /chemin/vers/dataset \
  --splits-dir splits \
  --sam2-ckpt /chemin/vers/sam2.pt \
  --model-ckpt model_phase3_best_ep15.pt \
  --n-images 5 \
  --out-dir runs/sam_eval_vote
```

La variante `eval_sam_m2.py` ajoute un fallback dense pour les pixels non couverts par les régions SAM2.

### YOLO-seg

Convertir des annotations COCO au format YOLO-seg :

```bash
python scripts/convert_coco_to_yolo.py --help
```

Puis évaluer une fusion YOLO-seg + FPN :

```bash
python scripts/eval_yolo_fusion.py \
  --dataset-root /chemin/vers/dataset \
  --splits-dir splits \
  --model-ckpt model_phase3_best_ep15.pt \
  --yolo-ckpt /chemin/vers/yolo_seg.pt \
  --split test \
  --out-dir runs/yolo_fusion
```

## Classes et métriques

Le modèle est configuré pour `11` classes. La classe `0` est ignorée dans le calcul de la loss et des métriques (`ignore_index=0`). Les métriques principales sont :

- **Pixel accuracy** : proportion de pixels correctement classés ;
- **IoU** : intersection-over-union calculée par classe ;
- **mIoU** : moyenne des IoU des classes évaluées.

## Reproductibilité

Les seeds et hyperparamètres sont définis dans les fichiers YAML. Pour reproduire une expérience, conservez la configuration générée dans le dossier de sortie ainsi que le checkpoint utilisé.

## Documentation complémentaire

Pour une description détaillée des modules, des choix d'architecture et des formats de sortie, consultez [`PROJECT_FILES_DESCRIPTION.md`](../PROJECT_FILES_DESCRIPTION.md).

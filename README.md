
# Projet PFE : Détection et Classification (YOLO + DualCrossViT)

Ce projet utilise **uv** pour la gestion des dépendances et du verrouillage des versions (`uv.lock`). Il permet de préparer des données via **YOLOv7** et d'entraîner un modèle **DualCrossViT**.


## Installation
Projet avec **UV**, utilisez la commande:

```bash
uv sync
```

#### Pour lancement local

Variables d'environnement dans [.env](.env), ou manuellement:

- bash:
```bash
export PYTHONPATH=src:external
```

- powershell:
```powershell
$env:PYTHONPATH="src;external"
```

> [!NOTE]
> Sur VSCode, configurations disponibles dans [`.vscode/launch.json`](.vscode/launch.json). Inclut `.env`.


## Pipelines

### Preparation

Copie le dataset (thorns ou genera) dans une arborescence normalisée:

```text
output_dir/           
    ├── original/         
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── segmented/        
        ├── class1/
        ├── class2/
        └── ...
```

#### Lancement

```bash
uv run src/pfe_crossvit_dual/preparation/preparation.py -d "thorns -i "raw/thorns" -o "data/thorns"
```

| Argument | Description | Défaut |
| :--- | :--- | :--- |
| `-d`, `--dataset` | Nom du dataset ("thorns" ou "genera") | `"thorns"` |
| `-i`, `--input` | Dossier contenant le dataset brut | `-` |
| `-o`, `--output` | Dossier où copier le dataset propre | `-` |

---
### Pretraitement

Utilise YOLOv7_ag pour segmenter les images dans les classes suivantes: \
[leaf, root, stem, flower, fruit, seed, BACKGROUND]

Pour chaque images, enregistre les segmentations ainsi que les patches labélisés au format `.pt`.

#### Lancement

```bash
uv run src/pfe_crossvit_dual/preprocessing/preprocessing.py -d "data/thorns -c
```

| Argument | Description | Défaut |
| :--- | :--- | :--- |
| `-d`, `--data_dir` | Dossier contenant le dataset propre | `-` |
| `-c`, `--clean` (optional) | Overwrite les poids existants, sinon skip | `False` |
| `-l`, `--limit` (optional) | (Debug only) limite max de fichiers | `None` |

---
### Entrainement

Pipeline d'entrainement du modèle DualCrossViT.

Tous les paramètres (entrainement, dataset, modèle) sont définis dans `config/parameters.yaml`.

**Modèles**:

- **DualCrossVitYolo**: prends en entrée l'image originale (branche large), des patches d'organe spécifiques produit par YOLOv7_ag, le masque de ségmentation ded la plante.

- **DualCrossVitRatio**: prends en entrée l'image originale (branche large), l'image segmentée (branche small), et les poids par patchs définis par une fonction dépendant du ratio de plante présente sur le patch.

#### Lancement

**Kaggle**

Uploader les datasets prétraités sur kaggle.

Utiliser le notebook suivant: [`notebooks/kaggle_run.ipynb`](notebooks/kaggle_run.ipynb).

Les paramètres d'entrainement sont définis dans le notebook dans un dict. \
[`notebooks/kaggle_helper.ipynb`](notebooks/kaggle_helper.ipynb) permet de transformer `parameters.yaml` en un dictionnaire pour le copier-coller.

**Local**

```bash
uv run src/pfe_crossvit_dual/training/training_pipeline.py -c "config/parameters.yaml"
```

| Argument | Description | Défaut |
| :--- | :--- | :--- |
| `-c`, `--config` | Fichier .yaml contenant les parametères | `config/parameters.yaml` |


#### Sortie

Les fichiers résultats sont disponibles dans le dossier `output/run_<i>`. \
Les fichiers sauvegardés sont:

- Le meilleur modèle `best_model_vit.pth`.
- Les logs d'entrainement `training_logs.txt`
- Les images de diagnostique d'attention dans `images/`

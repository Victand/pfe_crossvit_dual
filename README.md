
# Projet PFE : Détection et Classification (YOLO + DualCrossViT)

Ce projet utilise **uv** pour la gestion des dépendances et du verrouillage des versions (`uv.lock`). Il permet de préparer des données via **YOLOv7** et d'entraîner un modèle **DualCrossViT**.


Je dois changer des trucs genre les dossiers si ya plus de catégories ça va être trop lourd enpretraitement donc chiantos
## 1. Installation

Comme le projet contient un fichier `uv.lock`, utilisez la commande suivante pour synchroniser votre environnement :

```bash
uv sync
```

---

## 2. Détection et Préparation (`main_detection.py`)

Ce script gère le tri des images et la génération des poids de détection. 

**Note :** Si vos dossiers ne sont pas encore créés, la fonction `load_images` peut être appelée au début du script pour répartir les données du JSON vers les dossiers `train` et `val`.

### Lancement avec uv :
```bash
# Traitement complet avec nettoyage
uv run pretraitement.py --clean

# Test rapide sur une limite d'images
uv run pretraitement.py --limit 10
```
limit de 10 daans chaque sous dossier
---

## 3. Entraînement du Modèle (`train_vit.py`)

Une fois les images triées et les poids générés, ce script lance l'entraînement du Transformer.

### Lancement avec uv :
```bash
# Entraînement standard
uv run train_vic.py --batch 4 --lr 1e-4
batch de 4 pcq cpu
# Test rapide sur 100 samples avec visualisation
uv run train_vic.py --samples 100 --plot --batch 4
```

### Arguments disponibles :
| Argument | Description | Défaut |
| :--- | :--- | :--- |
| `-n`, `--samples` | Nombre d'images max à charger | `None` |
| `-e`, `--epochs` | Nombre d'époques | `7` |
| `-b`, `--batch` | Taille du batch | `32` |
| `--lr` | Learning rate | `1e-4` |
| `--plot` | Affiche une heatmap de contrôle avant le départ | `False` |

---

## Structure des Données
Le script génère et utilise l'arborescence suivante dans `DATA_DIR` :
```text
/created_data/
├── train/
│   └── original/
│       ├── epines/
│       └── no_epines/
└── val/
    └── original/
        ├── epines/
        └── no_epines/
```

---

## Sauvegarde
* Le meilleur modèle est sauvegardé sous : `saved/best_model_vit.pth`.
* L'état final est sauvegardé sous : `saved/final_model_vit.pth`.

### Note sur la fonction `load_images`
Si les dossiers de données ne sont pas encore créés ou sont vides, vous pouvez décommenter l'appel à `load_images()` dans le bloc `if __name__ == "__main__":` du script de détection pour forcer la redistribution des images à partir du fichier JSON.
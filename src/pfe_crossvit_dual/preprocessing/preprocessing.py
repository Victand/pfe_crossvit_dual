import json
import random
import argparse
import os
from pathlib import Path

from pfe_crossvit_dual.preprocessing.detect_perso import run_detect_perso
from pfe_crossvit_dual.constants.paths import DATA_DIR, YOLO_WEIGHTS


def load_images():
    CHEMIN_JSON = (
        "/mnt/2210B8B210B88E73/Documents/Ing 3/PFE/data/Data_Projet/data_propre.json"
    )
    DOSSIER_RACINE = Path("/data/created_data")

    # Création de l'arborescence
    for phase in ["train", "val"]:
        (DOSSIER_RACINE / phase / "original" / "epines").mkdir(
            parents=True, exist_ok=True
        )
        (DOSSIER_RACINE / phase / "original" / "no_epines").mkdir(
            parents=True, exist_ok=True
        )

    with open(CHEMIN_JSON, "r", encoding="utf-8") as fichier:
        donnees = json.load(fichier)

    # Séparation et mélange des classes
    liste_epines = [item for item in donnees if item["epine"] == 1]
    liste_sans_epines = [item for item in donnees if item["epine"] == 0]

    random.seed(42)
    random.shuffle(liste_epines)
    random.shuffle(liste_sans_epines)

    # Calcul du split 80/20
    coupure_epines = int(len(liste_epines) * 0.8)
    coupure_sans_epines = int(len(liste_sans_epines) * 0.8)


def full_process(limit=None, clean=False):
    for phase in ["train", "val"]:
        for classe in ["no_epines", "epines"]:
            dossier = os.path.join(DATA_DIR, phase, "original", classe)

            if os.path.isdir(dossier):
                print(f"\n--- Traitement : {phase} / {classe} ---")
                run_detect_perso(
                    weights=[YOLO_WEIGHTS],
                    source=str(dossier),
                    grid_size=64,
                    limit=limit,
                    clean_pt=clean,
                    skip_existing=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limite d'images")
    parser.add_argument("--clean", action="store_true", help="Nettoyage PT")

    args = parser.parse_args()
    full_process(limit=args.limit, clean=args.clean)

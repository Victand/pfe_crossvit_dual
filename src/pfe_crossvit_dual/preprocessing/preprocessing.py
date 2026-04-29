import argparse
import os

from pfe_crossvit_dual.preprocessing.detect_perso import run_detect_perso
from pfe_crossvit_dual.constants.paths import YOLO_WEIGHTS


def full_process(data_dir: str, limit=None, clean=False):
    classes = os.listdir(os.path.join(data_dir, "original"))
    print(classes)

    for classe in classes:
        dossier = os.path.join(data_dir, "original", classe)

        if os.path.isdir(dossier):
            print(f"\n--- Traitement : {classe} ---")
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
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to directory containing directory of original images",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limite d'images")
    parser.add_argument("--clean", action="store_true", help="Nettoyage PT")

    args = parser.parse_args()
    full_process(args.data_dir, limit=args.limit, clean=args.clean)

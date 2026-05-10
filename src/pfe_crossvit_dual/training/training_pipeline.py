import yaml
import torch

import torch.nn as nn
import os
import argparse
from pathlib import Path
import pprint

from pfe_crossvit_dual.constants.paths import OUTPUT_DIR
from pfe_crossvit_dual.training.dataset.data_helper import get_data
from pfe_crossvit_dual.training.model.model_loaders import (
    instanciate_dualcrossvit,
    load_training,
)
from pfe_crossvit_dual.training.utils.writersAndPlotters import save_training_graphs
from pfe_crossvit_dual.training.model.crossvit_kwargs import CROSSVIT_KWARGS_MAP
from pfe_crossvit_dual.training.train import train


def init_logs(save_path: Path, parameters, resume_path):
    """Initialisation du fichier de log et de l'historique"""
    log_file = os.path.join(save_path, "training_logs.txt")
    if resume_path:
        with open(log_file, "a") as f:
            f.write("\n\n=== Nouveaux Paramètres ===\n")
            f.write(pprint.pformat(parameters))
            f.write("\n\n=== Reprise de l'entraînement ===\n")
    else:
        with open(log_file, "w") as f:
            f.write(f"{save_path.name}\n")
            f.write("=== Paramètres ===\n")
            f.write(pprint.pformat(parameters))
            f.write("\n\n=== Début de l'entraînement ===\n")


def get_unique_save_path(base_dir="saved", prefix="run"):
    """
    Crée un dossier unique pour ne pas écraser les anciens entraînements.
    Exemple: saved/run_1, saved/run_2, etc.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        save_path = base_path / f"{prefix}_{i}"
        if not save_path.exists():
            save_path.mkdir(parents=True)
            return save_path
        i += 1


def training_pipeline(config):
    """Training Pipeline"""
    print("\n\n")
    # init dirs
    if config["resume_path"]:
        # Si tu reprends un entraînement, on sauvegarde dans le même dossier que le modèle chargé
        save_path = Path(config["resume_path"]).parent
        print(f"Resuming training at path {save_path}")
    else:
        # Sinon, on crée un nouveau dossier run_X
        save_path = get_unique_save_path(base_dir=OUTPUT_DIR, prefix="run")
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"New training {save_path.name}")

    # data
    print("Loading data...")
    img_size = CROSSVIT_KWARGS_MAP[config["model"]["crossvit"]]["img_size"]
    train_loader, val_loader, id_to_label = get_data(
        data_dir=config["dataset"]["data_dir"],
        img_size=img_size,
        batch_size=config["batch_size"],
        **config["dataset"],
    )

    # model
    print("Instanciating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instanciate_dualcrossvit(
        device=device,
        **config["model"],
        num_classes=len(id_to_label),
    )

    criterion = nn.CrossEntropyLoss()
    alphas = torch.tensor(config["alphas"]).to(device)

    if config["resume_path"] and os.path.isfile(config["resume_path"]):
        model, optimizer = load_training(config["resume_path"], model, config["lr"])
    else:
        optimizer = None

    # logs
    init_logs(save_path, config, config["resume_path"])

    # training loop
    print("Training model...")
    history, best_f1 = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        alphas,
        id_to_label,
        config["lr"],
        config["lr_factors"],
        config["epochs"],
        config["patience"],
        config["freeze"],
        config["unfreeze_schedule"],
        device,
        save_path,
    )

    # save training info
    save_training_graphs(history, save_path)
    with open(os.path.join(save_path, "training_logs.txt"), "a") as f:
        f.write("=== Entraînement terminé ===\n")
        f.write(f"Meilleure F1-score : {best_f1:.2f}%\n")

    print(
        f"\nEntraînement terminé. Meilleure Acc : {best_f1:.2f}%. Les logs et graphiques sont dans {save_path}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to yaml config",
    )

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.SafeLoader)

    training_pipeline(config["training"])

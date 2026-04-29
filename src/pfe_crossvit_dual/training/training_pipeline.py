import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import os
from pathlib import Path

from pfe_crossvit_dual.constants.paths import CONFIG, OUTPUT_DIR
from pfe_crossvit_dual.training.dataset.data_helper import get_data
from pfe_crossvit_dual.training.model.model_loaders import (
    instanciate_dualcrossvit,
    load_training,
)
from pfe_crossvit_dual.training.utils.writersAndPlotters import save_training_graphs
from pfe_crossvit_dual.training.model.crossvit_kwargs import crossvit_kwargs_map
from pfe_crossvit_dual.training.train import train


def init_logs(
    save_path, model_name, alphas, epochs, lr, batch_size, patience, resume_path
):
    """Initialisation du fichier de log et de l'historique"""
    log_file = os.path.join(save_path, "training_logs.txt")
    mode = "a" if resume_path else "w"
    with open(log_file, mode) as f:
        f.write("=== Début de l'entraînement ===\n")
        f.write(
            f"Modèle: {model_name} | Epochs: {epochs} | LR: {lr}|batch size {batch_size}|patience {patience}\n"
            f"stratégie {alphas.cpu()}\n\n"
        )


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
    # init dirs
    if config["resume_path"]:
        # Si tu reprends un entraînement, on sauvegarde dans le même dossier que le modèle chargé
        save_path = Path(config["resume_path"]).parent
    else:
        # Sinon, on crée un nouveau dossier run_X
        save_path = get_unique_save_path(base_dir=OUTPUT_DIR, prefix="run")
        save_path.mkdir(parents=True, exist_ok=True)

    # data
    img_size = crossvit_kwargs_map[config["model"]["crossvit"]]["img_size"]
    train_loader, val_loader, id_to_label = get_data(
        img_size=img_size,
        batch_size=config["batch_size"],
        **config["dataset"],
    )

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instanciate_dualcrossvit(
        device=device,
        **config["model"],
        num_classes=len(id_to_label),
    )
    optimizer = optim.Adam(
        params=model.parameters(), lr=config["lr"], weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    alphas = torch.tensor(config["alphas"]).to(device)

    if config["resume_path"] and os.path.isfile(config["resume_path"]):
        load_training(config["resume_path"], model, optimizer)

    # training loop
    init_logs(
        save_path,
        config["model"]["crossvit"],
        alphas,
        epochs=config["epochs"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        patience=config["patience"],
        resume_path=config["resume_path"],
    )
    history, best_acc = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        alphas,
        id_to_label,
        config["epochs"],
        config["patience"],
        device,
        save_path,
    )

    # save training info
    save_training_graphs(history, save_path)
    with open(os.path.join(save_path, "training_logs.txt"), "a") as f:
        f.write("=== Entraînement terminé ===\n")
        f.write(f"Meilleure Accuracy : {best_acc:.2f}%\n")

    print(
        f"\nEntraînement terminé. Meilleure Acc : {best_acc:.2f}%. Les logs et graphiques sont dans {save_path}."
    )


if __name__ == "__main__":
    config = yaml.load(open(CONFIG, "r"), Loader=yaml.SafeLoader)

    training_pipeline(config["training"])

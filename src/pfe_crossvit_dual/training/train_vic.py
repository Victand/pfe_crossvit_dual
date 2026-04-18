import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc


"""# Importations locales
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)"""

from pfe_crossvit_dual.training.model.dual_cross_vic import DualCrossVit
from pfe_crossvit_dual.training.dataset.dataset_vic import (
    DualInputDataset,
    prepare_dataloaders,
)
from pfe_crossvit_dual.training.model.model_loaders import (
    load_crossvit_pretrained_weights,
)

# Configuration globale
DATA_PATH = Path("/mnt/2210B8B210B88E73/Documents/Ing 3/PFE/data/created_data")
CLASSES = ("no_epines", "epines")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paramètres du modèle CrossViT
MODEL_NAME = "crossvit_15_224"
BRANCHES_IMG_SIZE = (224, 224)
PATCH_SIZE = (16, 16)
EMBED_DIM = (192, 384)
DEPTH = [[1, 5, 0], [1, 5, 0], [1, 5, 0]]
NUM_HEADS = (6, 6)
MLP_RATIO = (3.0, 3.0, 1.0)

STRATEGIE_ALPHAS = torch.tensor([1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 0.2]).to(DEVICE)


def instanciateDualCrossVit(model_name, device, lr, epochs, **model_kwargs):
    if "num_classes" not in model_kwargs:
        model_kwargs["num_classes"] = 2

    model = DualCrossVit(**model_kwargs)
    model = load_crossvit_pretrained_weights(model, model_name)
    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    return model, optimizer, scheduler, loss_fn


def get_data(n_samples=None, batch_size=16, num_workers=6):
    train_ds = DualInputDataset(
        data_dir=DATA_PATH,
        is_train=True,
        image_size=BRANCHES_IMG_SIZE,
        classes=CLASSES,
        paths=("original", "original"),
        use_yolo_weights=True,
        num_patches=8,
    )
    val_ds = DualInputDataset(
        data_dir=DATA_PATH,
        is_train=False,
        image_size=BRANCHES_IMG_SIZE,
        classes=CLASSES,
        paths=("original", "original"),
        use_yolo_weights=True,
        num_patches=8,
    )

    if n_samples:
        train_ds = Subset(train_ds, range(min(n_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(n_samples, len(val_ds))))

    return prepare_dataloaders(train_ds, val_ds, batch_size, num_workers)


def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_small, img_large, labels, weights in loader:
            x_small = x_small.to(DEVICE)
            img_large = img_large.to(DEVICE)
            labels = labels.to(DEVICE)
            weights = weights.to(DEVICE)

            preds = model(x_small, img_large, weights=weights, alpha=STRATEGIE_ALPHAS)
            loss = criterion(preds, labels)
            val_loss += loss.item()

            _, predicted = torch.max(preds.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(loader)

    # Calcul des métriques avec scikit-learn
    acc = accuracy_score(all_labels, all_preds) * 100
    prec = (
        precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    )
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    return avg_loss, acc, prec, rec, f1


def save_training_graphs(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Graphique de la Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", marker="o")
    ax1.set_title("Évolution de la Perte (Loss)")
    ax1.set_xlabel("Époques")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Graphique des Métriques
    ax2.plot(epochs, history["val_acc"], label="Accuracy", marker="s")
    ax2.plot(epochs, history["val_prec"], label="Precision", marker="^")
    ax2.plot(epochs, history["val_rec"], label="Recall", marker="v")
    ax2.plot(epochs, history["val_f1"], label="F1-Score", marker="d")
    ax2.set_title("Évolution des Métriques de Validation")
    ax2.set_xlabel("Époques")
    ax2.set_ylabel("Score (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    save_path = Path(save_dir) / "training_metrics_graph.png"
    plt.savefig(save_path)
    plt.close()
    print(f" > Graphiques sauvegardés dans : {save_path}")


def debug_full_diagnostic(
    x_small, img_large, weights, model, labels, classes, epoch, save_dir="saved/images"
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        _ = model(x_small, img_large, weights=weights, alpha=STRATEGIE_ALPHAS)
        attn = model.blocks[-1].blocks[1][-1].attn.last_attn_map

        if attn is None:
            return

        attn = attn.mean(dim=1)
        cls_attn = attn[0, 0, 1:]
        grid_size = int(cls_attn.shape[-1] ** 0.5)
        vis = cls_attn.reshape(grid_size, grid_size).cpu().numpy()

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_vis = img_large[0].permute(1, 2, 0).cpu().numpy()
    img_vis = np.clip(std * img_vis + mean, 0, 1)

    fig = plt.figure(figsize=(22, 10))
    gs = plt.GridSpec(2, 4, width_ratios=[1.2, 1.2, 0.8, 0.8])

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(img_vis)
    if weights is not None:
        h_map = (
            (weights[0] * STRATEGIE_ALPHAS[: weights.shape[1]].view(-1, 1, 1))
            .sum(dim=0)
            .cpu()
            .numpy()
        )
        ax1.imshow(h_map, cmap="jet", alpha=0.3)
    ax1.set_title(f"INPUT (Image + Heatmap)\nClasse: {classes[labels[0]]}")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(img_vis)
    ax2.imshow(
        vis,
        cmap="viridis",
        alpha=0.7,
        extent=(0, img_vis.shape[1], img_vis.shape[0], 0),
        interpolation="nearest",
    )
    ax2.set_title("OÙ LE MODÈLE REGARDE\n(Self-Attention Branche Large)")
    ax2.axis("off")

    inner_gs = gs[:, 2:].subgridspec(4, 2)
    patches = x_small[0]
    for i in range(min(len(patches), 8)):
        ax_p = fig.add_subplot(inner_gs[i // 2, i % 2])
        p_img = patches[i].permute(1, 2, 0).cpu().numpy()
        p_img = np.clip(std * p_img + mean, 0, 1)
        ax_p.imshow(p_img)
        ax_p.axis("off")
        ax_p.set_title(f"Patch {i + 1}", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"diag_epoch_{epoch}.png"))
    plt.close()


def train(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    batch_size=16,
    save_dir="saved",
    resume_path=None,
    patience=5,
):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_acc = 0.0
    no_imporve = 0

    # --- LOGIQUE DE REPRISE ---
    if resume_path and os.path.isfile(resume_path):
        print(f" > Reprise de l'entraînement depuis : {resume_path}")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get("val_acc", 0.0)
        print(
            f" > Reprise à l'époque {start_epoch + 1} avec une Accuracy de {best_acc:.2f}%"
        )
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialisation du fichier de log et de l'historique
    log_file = save_path / "training_logs.txt"
    mode = "a" if resume_path else "w"
    with open(log_file, mode) as f:
        f.write("=== Début de l'entraînement ===\n")
        f.write(
            f"Modèle: {MODEL_NAME} | Époques: {epochs} | LR: {lr}|batch size {batch_size}|patience {patience}\n"
            f"stratégie {STRATEGIE_ALPHAS}\n\n"
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_prec": [],
        "val_rec": [],
        "val_f1": [],
    }

    print(f"Lancement de l'entraînement sur {DEVICE}...")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Époque {epoch + 1}/{epochs}")

        for x_small, img_large, labels, weights in pbar:
            x_small = x_small.to(DEVICE)
            img_large = img_large.to(DEVICE)
            labels = labels.to(DEVICE)
            weights = weights.to(DEVICE)

            optimizer.zero_grad()
            preds = model(x_small, img_large, weights=weights, alpha=STRATEGIE_ALPHAS)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        gc.collect()
        torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader)

        # Évaluation complète
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion
        )

        # Enregistrement dans l'historique
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_prec"].append(val_prec)
        history["val_rec"].append(val_rec)
        history["val_f1"].append(val_f1)

        # Affichage console
        print(
            f"\nÉpoque {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        print(
            f"Metrics -> Acc: {val_acc:.2f}% | Prec: {val_prec:.2f}% | Rec: {val_rec:.2f}% | F1: {val_f1:.2f}%"
        )

        # Écriture dans le fichier de log
        with open(log_file, "a") as f:
            f.write(f"Époque {epoch + 1}/{epochs}\n")
            f.write(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}\n")
            f.write(
                f"Accuracy: {val_acc:.2f}% | Precision: {val_prec:.2f}% | Recall: {val_rec:.2f}% | F1-Score: {val_f1:.2f}%\n"
            )
            f.write("-" * 50 + "\n")

        # Sauvegarde du diagnostic visuel
        x_s, x_l, lbl, w = next(iter(val_loader))
        debug_full_diagnostic(
            x_s.to(DEVICE),
            x_l.to(DEVICE),
            w.to(DEVICE),
            model,
            lbl,
            CLASSES,
            epoch + 1,
            save_dir=save_path / "images",
        )

        # Sauvegarde du meilleur modèle
        if val_acc > best_acc:
            no_imporve = 0
            best_acc = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "alpha_used": STRATEGIE_ALPHAS.cpu(),
            }
            torch.save(checkpoint, save_path / "best_model_vit.pth")
            print(" > Nouveau record ! Modèle sauvegardé.")
            with open(log_file, "a") as f:
                f.write(
                    f"*** Nouveau meilleur modèle sauvegardé (Acc: {best_acc:.2f}%) ***\n\n"
                )

        else:
            no_imporve += 1
            if no_imporve >= patience:
                print("early stop")
                with open(log_file, "a") as f:
                    f.write(
                        f"Early stop à l'epoch: {epoch}, pas d'améloration depuis {patience} epochs"
                    )
                break

    # Fin de l'entraînement
    torch.save(model.state_dict(), save_path / "final_model_vit.pth")
    plt.close("all")
    save_training_graphs(history, save_dir)

    with open(log_file, "a") as f:
        f.write("=== Entraînement terminé ===\n")
        f.write(f"Meilleure Accuracy : {best_acc:.2f}%\n")

    print(
        f"\nEntraînement terminé. Meilleure Acc : {best_acc:.2f}%. Les logs et graphiques sont dans {save_dir}."
    )


def get_unique_save_dir(base_dir="saved", prefix="run"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraînement DualCrossViT avec guidage YOLO"
    )
    parser.add_argument(
        "-n", "--samples", type=int, default=None, help="Nombre de samples pour le test"
    )
    parser.add_argument("-e", "--epochs", type=int, default=7, help="Nombre d'époques")
    parser.add_argument("-b", "--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--resume", type=str, default=None, help="Chemin vers best_model_vit.pth"
    )
    args = parser.parse_args()

    if args.resume:
        # Si tu reprends un entraînement, on sauvegarde dans le même dossier que le modèle chargé
        run_save_dir = Path(args.resume).parent
    else:
        # Sinon, on crée un nouveau dossier run_X
        run_save_dir = get_unique_save_dir(base_dir="saved", prefix="run")
    # Préparation des données
    ld_train, ld_val = get_data(
        n_samples=args.samples, batch_size=args.batch, num_workers=3
    )

    # Initialisation du modèle
    model, _, _, _ = instanciateDualCrossVit(
        model_name=MODEL_NAME,
        device=DEVICE,
        lr=args.lr,
        epochs=args.epochs,
        img_size=BRANCHES_IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=len(CLASSES),
    )

    # Lancement
    train(
        model,
        ld_train,
        ld_val,
        args.epochs,
        args.lr,
        batch_size=args.batch,
        save_dir=run_save_dir,
        resume_path=args.resume,
        patience=10,
    )

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Subset
from tqdm import tqdm

# Importations locales
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from pfe_crossvit_dual.training.old.dataset_youenn import (
    DualInputDataset,
    prepare_dataloaders,
    find_two_samples,
    plot_samples_with_weights,
)
from pfe_crossvit_dual.training.model.model_loaders import instanciateDualCrossVit

# Configuration globale
DATA_PATH = Path("/mnt/2210B8B210B88E73/Documents/Ing 3/PFE/data/created_data")
CLASSES = ("no_epines", "epines")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paramètres du modèle CrossViT
MODEL_NAME = "crossvit_15_224"
BRANCHES_IMG_SIZE = (240, 224)
PATCH_SIZE = (12, 16)
EMBED_DIM = (192, 384)
DEPTH = [[1, 5, 0], [1, 5, 0], [1, 5, 0]]
NUM_HEADS = (6, 6)
MLP_RATIO = (3.0, 3.0, 1.0)

# Stratégie de poids YOLO : [Fond, leaf, root, STEM, flower, fruit, seed]
STRATEGIE_ALPHAS = torch.tensor([0.1, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0]).to(DEVICE)


def get_data(n_samples=None, batch_size=32, num_workers=6):
    """Prépare les datasets et les loaders"""
    train_ds = DualInputDataset(
        data_dir=DATA_PATH,
        is_train=True,
        image_size=BRANCHES_IMG_SIZE,
        classes=CLASSES,
        paths=("original", "original"),
        use_yolo_weights=True,
    )
    val_ds = DualInputDataset(
        data_dir=DATA_PATH,
        is_train=False,
        image_size=BRANCHES_IMG_SIZE,
        classes=CLASSES,
        paths=("original", "original"),
        use_yolo_weights=True,
    )

    if n_samples:
        train_ds = Subset(train_ds, range(min(n_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(n_samples, len(val_ds))))

    return prepare_dataloaders(train_ds, val_ds, batch_size, num_workers)


def validate(model, loader, criterion):
    """Évalue le modèle sur l'ensemble de validation"""
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for img_orig, img_seg, labels, weights in loader:
            img_orig, img_seg, labels = (
                img_orig.to(DEVICE),
                img_seg.to(DEVICE),
                labels.to(DEVICE),
            )
            weights = (
                weights.to(DEVICE).squeeze(1)
                if weights.dim() == 5
                else weights.to(DEVICE)
            )

            preds = model(img_orig, img_seg, weights=weights, alpha=STRATEGIE_ALPHAS)
            loss = criterion(preds, labels)

            val_loss += loss.item()
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return val_loss / len(loader), 100 * correct / total


def train(model, train_loader, val_loader, epochs, lr, save_dir="saved"):
    """Boucle d'entraînement avec sauvegarde du meilleur modèle"""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Création du dossier de sauvegarde
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    print(f"Lancement de l'entraînement sur {DEVICE}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for img_orig, img_seg, labels, weights in pbar:
            img_orig, img_seg, labels = (
                img_orig.to(DEVICE),
                img_seg.to(DEVICE),
                labels.to(DEVICE),
            )
            weights = (
                weights.to(DEVICE).squeeze(1)
                if weights.dim() == 5
                else weights.to(DEVICE)
            )

            optimizer.zero_grad()
            preds = model(img_orig, img_seg, weights=weights, alpha=STRATEGIE_ALPHAS)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        avg_val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"\nEpoch {epoch + 1} : Val Acc = {val_acc:.2f}%")

        # --- SAUVEGARDE DU MEILLEUR MODÈLE ---
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "alpha_used": STRATEGIE_ALPHAS.cpu(),
            }
            torch.save(checkpoint, save_path / "best_model_vit.pth")
            print(
                f" > Nouveau record ! Modèle sauvegardé dans {save_path}/best_model_vit.pth"
            )

    # Sauvegarde finale (dernier état)
    torch.save(model.state_dict(), save_path / "final_model_vit.pth")
    print(f"\nEntraînement terminé. Meilleure Acc : {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraînement DualCrossViT avec guidage YOLO"
    )
    parser.add_argument(
        "-n", "--samples", type=int, default=None, help="Nombre de samples pour le test"
    )
    parser.add_argument("-e", "--epochs", type=int, default=7, help="Nombre d'Epochs")
    parser.add_argument("-b", "--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--plot", action="store_true", help="Afficher la heatmap avant de lancer"
    )
    args = parser.parse_args()

    # Préparation des données
    ld_train, ld_val = get_data(
        n_samples=args.samples, batch_size=args.batch, num_workers=6
    )

    # Optionnel : Visualisation de contrôle
    if args.plot:
        ds_source = (
            ld_train.dataset.dataset
            if isinstance(ld_train.dataset, Subset)
            else ld_train.dataset
        )
        samples = find_two_samples(ds_source.samples)
        plot_samples_with_weights(
            samples, CLASSES, ("original", "original"), STRATEGIE_ALPHAS.cpu()
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
    train(model, ld_train, ld_val, args.epochs, args.lr)

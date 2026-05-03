import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc

from pfe_crossvit_dual.training.utils.diagnostic import debug_full_diagnostic


LR_FACTORS = {
    "head": 1.0,
    "norm": 0.3,
    "fusion_blocks": 0.1,
    "positional": 0.05,
    "patch_embed": 0.02,
}


def validate(model, loader, criterion, alphas, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_small, img_large, weights, labels in tqdm(loader, desc="val"):
            x_small = x_small.to(device)
            img_large = img_large.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            preds = model(x_small, img_large, weights=weights, alpha=alphas)
            loss = criterion(preds, labels)
            val_loss += loss.item()

            _, predicted = torch.max(preds.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(loader)

    # Calcul des métriques avec scikit-learn
    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    return avg_loss, acc, prec, rec, f1


def build_optimizer(model, base_lr=3e-4):
    param_groups = []
    # Head
    param_groups.append(
        {
            "params": [p for head in model.head for p in head.parameters() if p.requires_grad],
            "lr": base_lr * LR_FACTORS["head"],
        }
    )
    # Norm
    param_groups.append(
        {
            "params": [p for n in model.norm for p in n.parameters() if p.requires_grad],
            "lr": base_lr * LR_FACTORS["norm"],
        }
    )
    # Fusion blocks
    param_groups.append(
        {
            "params": [p for blk in model.blocks for p in blk.parameters() if p.requires_grad],
            "lr": base_lr * LR_FACTORS["fusion_blocks"],
        }
    )
    # Positional stuff
    param_groups.append(
        {
            "params": [p for p in model.pos_embed if p.requires_grad]
            + [p for p in model.cls_token if p.requires_grad],
            "lr": base_lr * LR_FACTORS["positional"],
        }
    )
    # Patch embedding
    param_groups.append(
        {
            "params": [p for pe in model.patch_embed for p in pe.parameters() if p.requires_grad],
            "lr": base_lr * LR_FACTORS["patch_embed"],
        }
    )
    return torch.optim.AdamW(param_groups, weight_decay=1e-4)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    alphas,
    id_to_label,
    base_lr,
    epochs,
    patience,
    freeze,
    unfreeze_schedule,
    device,
    save_path,
):
    log_path = os.path.join(save_path, "training_logs.txt")
    img_dir = os.path.join(save_path, "images")
    os.makedirs(img_dir, exist_ok=True)

    if optimizer is None:
        optimizer = build_optimizer(model, base_lr)

    start_epoch = 0
    best_f1 = 0.0
    no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_prec": [],
        "val_rec": [],
        "val_f1": [],
    }

    if freeze:
        print(f"unfreezing stage: {0}")
        model.set_unfreeze_stage(0)

    print(f"Lancement de l'entraînement sur {device}...")

    for epoch in range(start_epoch, epochs):
        model.train()

        if freeze:
            stage = -1
            for k, v in unfreeze_schedule.items():
                if v == epoch + 1:
                    stage = k
            if stage >= 0:
                print(f"unfreezing stage: {stage}")
                model.set_unfreeze_stage(stage)
                optimizer = build_optimizer(model, base_lr)

        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{epoch + 1}/{epochs}] train")

        for x_small, img_large, weights, labels in pbar:
            x_small = x_small.to(device, non_blocking=True)
            img_large = img_large.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(x_small, img_large, weights=weights, alpha=alphas)
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
            model, val_loader, criterion, alphas, device
        )

        # Enregistrement dans l'historique
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_prec"].append(val_prec)
        history["val_rec"].append(val_rec)
        history["val_f1"].append(val_f1)

        # Affichage console
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"Metrics -> Acc: {val_acc:.2f}% | Prec: {val_prec:.2f}% | Rec: {val_rec:.2f}% | F1: {val_f1:.2f}%"
        )

        # Écriture dans le fichier de log
        with open(log_path, "a") as f:
            f.write(f"Epoch {epoch + 1}/{epochs}\n")
            f.write(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}\n")
            f.write(
                f"Accuracy: {val_acc:.2f}% | Precision: {val_prec:.2f}% | Recall: {val_rec:.2f}% | F1-Score: {val_f1:.2f}%\n"
            )
            f.write("-" * 50 + "\n")

        # Sauvegarde du diagnostic visuel
        x_s, x_l, w, lbl = next(iter(val_loader))
        diagnostic_fp = os.path.join(img_dir, f"diag_epoch_{epoch + 1}.png")
        debug_full_diagnostic(
            x_s.to(device),
            x_l.to(device),
            w.to(device),
            model,
            alphas,
            lbl,
            id_to_label,
            diagnostic_fp,
        )

        # Sauvegarde du meilleur modèle
        if val_f1 > best_f1:
            no_improve = 0
            best_f1 = val_f1
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "alpha_used": alphas.cpu(),
            }
            torch.save(checkpoint, save_path / "best_model.pth")
            print(" > Nouveau record ! Modèle sauvegardé.\n")
            with open(log_path, "a") as f:
                f.write(
                    f"*** Nouveau meilleur modèle sauvegardé (F1-score: {best_f1:.2f}%) ***\n\n"
                )

        else:
            no_improve += 1
            print(f"no improvement [{no_improve}/{patience}]")
            if no_improve >= patience:
                print("early stop")
                with open(log_path, "a") as f:
                    f.write(
                        f"Early stop à l'epoch: {epoch}, pas d'améloration depuis {patience} epochs"
                    )
                break

    return history, best_f1

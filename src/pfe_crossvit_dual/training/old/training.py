import torch
from torchmetrics.classification import BinaryF1Score
from pathlib import Path
import time
import numpy as np
import datetime

from pfe_crossvit_dual.training.old.IoULoss import IoUConstrainedLoss
from pfe_crossvit_dual.training.utils.writersAndPlotters import *
from pfe_crossvit_dual.training.model.attentionRollout import (
    get_trainable_heatmap,
    get_iou_training,
)


def dual_eval(
    model,
    data_loader,
    loss_fn,
    device,
    patch_size,
    plot_pounderation=False,
    debug=False,
    failed_preds=0,
):
    """
    Computes evaluation for our custom dual input model

    Args:
        failed_preds > 0 if you want to show some fail examples.

    Returns :
        avg_val_loss and accuracy on dataset
    """
    model.eval()
    total_val_loss = 0
    total_iou = 0

    # true_positives, false_positives, false_negatives = 0,0,0
    f1_metric = BinaryF1Score().to(device)

    count_correct_pred = 0
    total_samples = 0

    with torch.no_grad():
        failed_imgs = []

        for batch_id, (original_image, segmented_image, label_gt, weights) in enumerate(
            data_loader
        ):
            original_image = original_image.to(device)
            segmented_image = segmented_image.to(device)
            label_gt = label_gt.to(device)
            weights = weights.to(device)

            logits = model(original_image, segmented_image, weights)

            if plot_pounderation and (weights.numel() != 0) and batch_id == 0:
                plot_weight_example(segmented_image, weights, num_examples=1, patch_size=patch_size)

            val_loss = loss_fn(logits, label_gt)
            total_val_loss += val_loss.item()

            # If Loss function is our custom IouLoss then we record IoU metric
            if hasattr(loss_fn, "lambda_iou"):
                heatmap = get_trainable_heatmap(model, device)
                iou = get_iou_training(model, heatmap, segmented_image)
                total_iou += iou

            # argmax
            _, predicted_class = torch.max(logits, dim=1)

            # debug
            if debug and batch_id == 0:
                print(f"Vrais Labels (GT) : {label_gt.cpu().numpy()}")
                print(f"Predicted : {predicted_class.cpu().numpy()}")
                print(f"Logits (Scores)   : {logits[0].cpu().numpy()}")

            # To store failure examples if needed : if failed_preds > 0
            i = 0
            batch_size = len(original_image)
            while len(failed_imgs) < failed_preds and i < batch_size:
                if predicted_class[i] != label_gt[i]:
                    failed_imgs.append(
                        (
                            original_image[i].cpu(),
                            segmented_image[i].cpu(),
                            predicted_class[i].cpu(),
                            label_gt[i].cpu(),
                        )
                    )
                i += 1

            """true_positives += ((label_gt==1) & (predicted_class == 1)).sum().item()
            false_positives += ((label_gt==0) & (predicted_class == 1)).sum().item()
            false_negatives += ((label_gt==1) & (predicted_class == 0)).sum().item()"""
            f1_metric.update(predicted_class, label_gt)

            count_correct_pred += (predicted_class == label_gt).sum().item()
            total_samples += label_gt.size(0)

    if failed_preds > 0:
        if len(failed_imgs) == 0:
            print("Aucune mauvaise prédiction ! Goatesque ?!")
        else:
            classes = data_loader.dataset.classes
            mean = data_loader.dataset.mean
            std = data_loader.dataset.std
            show_tensors(failed_imgs, classes, mean, std)

    avg_val_loss = total_val_loss / len(data_loader)
    accuracy = 100 * count_correct_pred / total_samples
    avg_iou = total_iou / len(data_loader)

    """epsilon = 1e-7
    f1_score = 100 * (2*true_positives)/(false_positives+false_negatives+2*true_positives+epsilon)"""
    f1 = 100 * f1_metric.compute()
    f1 = f1.cpu()

    return avg_val_loss, accuracy, f1, avg_iou


def train_one_epoch(
    model,
    epoch_idx,
    epochs,
    best_val_loss,
    data_loader_train,
    data_loader_val,
    device,
    loss_fn,
    optimizer,
    scheduler,
    patch_size,
    pounderation,
    save,
    output_dir,
    log_file,
    args=None,
):
    model.train()
    total_train_loss = 0
    write_in_logs(log_file, message=f"\nEpoch {epoch_idx + 1}/{epochs}")

    for original_image, segmented_image, label_gt, weights in data_loader_train:
        original_image = original_image.to(device)
        segmented_image = segmented_image.to(device)
        label_gt = label_gt.to(device)
        weights = weights.to(device)

        label_pred = model(original_image, segmented_image, weights)

        train_loss = loss_fn(label_pred, label_gt)

        optimizer.zero_grad()
        train_loss.backward(retain_graph=True)

        optimizer.step()

        total_train_loss += train_loss.item()

    scheduler.step()
    avg_train_loss = total_train_loss / len(data_loader_train)

    # We want to plot for examples of weights on patches for the first epoch only
    plot_ponderation = False if epoch_idx > 0 else pounderation

    avg_val_loss, accuracy, f1_score, avg_iou = dual_eval(
        model, data_loader_val, loss_fn, device, patch_size, plot_ponderation
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        write_in_logs(log_file, message=f"\nBest val loss yet : {best_val_loss}")
        if save and (save == "BEST") and output_dir:
            torch.save(model.state_dict(), Path(output_dir) / "best_model.pth")

    if save and (save == "LAST") and output_dir:
        torch.save(model.state_dict(), Path(output_dir) / "last_model.pth")

    write_in_logs(
        log_file,
        message=f"\nAvg Train Loss: {avg_train_loss:.4f}.\nAvg Val Loss: {avg_val_loss:.4f}\nVal Accuracy: {accuracy:.2f}%\nF1-Score: {f1_score:.2f}%\n",
    )
    if avg_iou is not None:
        write_in_logs(log_file, message=f"Avg IoU: {avg_iou * 100:.4f}%.\n")

    return best_val_loss, avg_train_loss, avg_val_loss, accuracy, f1_score, avg_iou


def train(
    model,
    epochs,
    data_loader_train,
    data_loader_val,
    device,
    loss_fn,
    optimizer,
    scheduler,
    patch_size=16,
    pounderation=False,
    args=None,
    plot_loss=False,
    save=None,
    maskcaptor=None,
    output_dir="src\savedModels",
    log_file=None,
):
    """
    Training loop with clean interupt managed.

    Args:

        plot_loss (bool) : if you want to plot losses and accuracy evolution through epochs
        patch_size (int) : necessary if you want to plot the weights in case of pounderation
        save (str) : if need to save model choose 'LAST' or 'BEST' as strategy and precise output_dir
        output_dir (str):

    Returns:
        (best_val_loss, history):
            best val loss value found and {'train_loss': [], 'val_loss': [], 'val_acc': [], 'f1': []} list of values per epoch
    """
    write_in_logs(log_file, message=f"\nStart training for {epochs} epochs\n")

    start_time = time.time()

    best_val_loss = np.inf

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "f1": [], "iou": []}

    epoch_idx = 0

    handle = None
    if maskcaptor is not None:
        # maskcaptor captures the mask to computes IOU with our custom loss function.
        handle = model.register_forward_hook(maskcaptor.hook_fn)
        loss_fn = IoUConstrainedLoss(model, maskcaptor, lambda_iou=0.1, device=device)

    try:
        for epoch_idx in range(epochs):
            best_val_loss, avg_train_loss, avg_val_loss, accuracy, f1_score, avg_iou = (
                train_one_epoch(
                    model,
                    epoch_idx,
                    epochs,
                    best_val_loss,
                    data_loader_train,
                    data_loader_val,
                    device,
                    loss_fn,
                    optimizer,
                    scheduler,
                    patch_size,
                    pounderation,
                    save,
                    output_dir,
                    log_file,
                    args,
                )
            )

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(accuracy)
            history["f1"].append(f1_score)
            history["iou"].append(avg_iou)

    # Clean exit
    except KeyboardInterrupt:
        print("[Warning] Training interrupted by user.")

        if args and args.output_dir:
            write_in_logs(log_file, message="\nSaving emergency checkpoint...")
            ckpt_path = Path(args.output_dir) / f"interrupted_epoch_{epoch_idx}.pth"
            torch.save(
                {
                    "epoch": epoch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )

            write_in_logs(log_file, message=f"\n[Info] Emergency checkpoint saved : {ckpt_path}")

    finally:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        history["time"] = total_time

        write_in_logs(log_file, message=f"\nTraining time: {total_time_str}\n")

        if plot_loss:
            plot_train_metrics(history)

        if handle is not None:
            handle.remove()

    return best_val_loss, history

from torch import Tensor
import numpy as np
import datetime
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
import os


def write_in_logs(log_file, message, ammend=True):
    if log_file:
        mode = "a" if ammend else "w"
        with open(log_file, mode) as log_f:
            log_f.write(message)


def save_training_graphs(history, save_dir):
    plt.close("all")
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
    save_path = os.path.join(save_dir, "training_metrics_graph.png")
    plt.savefig(save_path)
    plt.close()
    print(f" > Graphiques sauvegardés dans : {save_path}")


def plot_train_metrics(history, title_infos=["time"], skip_keys=["iou"]):
    """
    Plot evolution through epochs of various metrics contained in history dict.
    For additional infos to show in title, put the keys in title_infos.
    """
    metrics_keys = list(history.keys())
    num_metrics = len(metrics_keys) - len(title_infos) - len(skip_keys)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(8, 12))

    colors = ["c", "m", "y", "r", "g", "b"]
    n_colors = len(colors)

    infos = ""
    i = 0
    for k in metrics_keys:
        if k in skip_keys:
            continue

        if k in title_infos:
            info = history[k]

            if k == "time":
                info = str(datetime.timedelta(seconds=int(info)))

            infos += f"\nTotal execution time : {info}"
            continue

        metric = history[k]
        ax = axes[i]

        ax.plot(metric, label=k, marker="o", linestyle="-", color=colors[i % n_colors])
        ax.set_title(f"{k}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        i += 1

    fig.suptitle(f"Resultats de l'entrainement.{infos}\n")
    plt.tight_layout()
    plt.show()


def plot_one_metric_multiple_configs(metric_name, configs_dict):
    """
    Plot evolution through epochs of a metric for every configs in dict.
    """
    configs_keys = list(configs_dict.keys())

    fig, ax = plt.subplots()

    colors = ["c", "m", "y", "r", "g", "b"]
    n_colors = len(colors)

    for i, k in enumerate(configs_keys):
        metric = configs_dict[k]["history"][metric_name]
        ax.plot(metric, label=k, marker="o", linestyle="-", color=colors[i % n_colors])

    ax.set_title(f"{metric_name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    fig.suptitle(f"{metric_name}\n")
    plt.tight_layout()
    plt.show()


def show_tensors(
    tensor_images, classes, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    """
    tensor_images = [(original_img,segmented_img, label_predit, label_gt),...]
    """
    n = len(tensor_images)
    fig, axes = plt.subplots(n, 2)

    for i, sample in enumerate(tensor_images):
        for j, tensor_img in enumerate(sample[:2]):
            img = tensor_img.clone().cpu()
            # (C, H, W) -> (H, W, C)
            img = img.permute(1, 2, 0).numpy()

            # Unnormalize
            mean = np.array(mean)
            std = np.array(std)
            img = std * img + mean

            img = np.clip(img, 0, 1)

            axes[i, j].imshow(img)

        axes[i, 0].set(title=f"Label prédit = {classes[sample[2]]}")
    fig.suptitle("Echantillons d'échecs de prédictions.")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_weight_example(
    segmented_images_batch: tuple[Tensor, ...],
    weights: Tensor,
    num_examples: int,
    patch_size: int,
    images_name=None,
    add_infos: str = "",
):
    """
    Shows the weights applied to the image, should be use in validation.
    We assume that this image is a square in the case of pounderation (Project Part 3).
    We assume that the image has not be normalized in transforms.
    """
    fig, axes = plt.subplots(1, num_examples, figsize=(5 * num_examples, 5))
    if num_examples == 1:
        axes = [axes]

    img_size = segmented_images_batch[0].shape[1]

    for i in range(num_examples):
        seg_img = segmented_images_batch[i].clone().cpu()
        img = seg_img.permute(1, 2, 0).numpy()

        img = np.clip(img, 0, 1)

        patches_weights = weights[i].clone().cpu().squeeze()

        grid_size = img_size // patch_size
        # NOTE NumPy remplit d'abord la dernière dimension. Si les poids semblent "tournés" de 90° lors de l'affichage,
        # il faut regarder du côté de .transpose() ou changer l'ordre du .view().
        w_grid = patches_weights.view(grid_size, grid_size).numpy()
        # Overlay RGBA to show weight with opacity
        overlay = np.zeros((grid_size, grid_size, 4))
        # Red color
        overlay[..., 0] = 1.0
        # Normalize to be between 0 and 1 to affect alpha
        if w_grid.max() > 0:
            overlay[..., 3] = (w_grid / w_grid.max()) * 0.7

        axes[i].imshow(img)
        axes[i].imshow(
            overlay, extent=(0, img_size, img_size, 0), interpolation="nearest"
        )
        if images_name is not None:
            axes[i].set_title(f"Image Id : {images_name[i]}")
        axes[i].axis("off")

    fig.suptitle(f"Patch weights visualization\n{add_infos}")
    plt.tight_layout()
    plt.show()


def compare_functions_for_patches_ponderation(
    dataset, functions: dict, num_images: int = 1
):
    """
    Compare different functions to create weights for patches.
    Args:
        functions {name:function}: give a name to your function to plot
    """
    imgs = []

    for sample in dataset.samples[:num_images]:
        _, segmented_image_p, _ = sample
        image_name = segmented_image_p.stem
        segmented_image = Image.open(segmented_image_p).convert("RGB")
        segmented_image_tensor = TF.to_tensor(
            TF.center_crop(
                TF.resize(segmented_image, int(dataset.image_size[1] * 1.14)),
                dataset.image_size[1],
            )
        )
        imgs.append((segmented_image_tensor, image_name))

    imgs_and_names = [list(row) for row in zip(*imgs)]

    for f_name, f in functions.items():
        weights = []
        for i in range(len(imgs)):
            weights.append(
                dataset.patches_weights(imgs[i][0], dataset.patch_size[1], f)
            )
        plot_weight_example(
            imgs_and_names[0],
            weights,
            num_images,
            dataset.patch_size[1],
            imgs_and_names[1],
            f"Function : {f_name}\n",
        )

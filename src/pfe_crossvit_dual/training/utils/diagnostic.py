import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pfe_crossvit_dual.training.dataset.dual_input_dataset import (
    IMGNET_MEAN,
    IMGNET_STD,
)


def tensor_to_img(x):
    # to numpy
    arr = x.permute(1, 2, 0).cpu().numpy()
    # unnormalize
    mean, std = np.array(IMGNET_MEAN), np.array(IMGNET_STD)
    arr = np.clip(std * arr + mean, 0, 1)
    return arr


def get_attention_grid(model, x_small, x_large, weights, alphas):
    model.eval()
    with torch.no_grad():
        _ = model(x_small, x_large, weights=weights, alpha=alphas)
    attn = model.blocks[-1].blocks[1][-1].attn.last_attn_map
    attn = attn.mean(dim=1)
    cls_attn = attn[0, 0, 1:]

    grid_size = int(cls_attn.shape[-1] ** 0.5)
    attn_grid = cls_attn.reshape(grid_size, grid_size).cpu().numpy()

    return attn_grid


def plot_yolo_patches(axs, patches, n_patches):
    for i in range(n_patches):
        axs[i].imshow(patches[i])
        axs[i].axis("off")
        axs[i].set_title(f"Patch {i + 1}", fontsize=8)


def plot_segmented_img(ax, img_seg):
    ax.imshow(img_seg)
    ax.set_title("Segmented image")
    ax.axis("off")


def plot_input(ax, img_large, weights, alphas, label):
    ax.imshow(img_large)
    if weights is not None:
        h_map = (weights * alphas[: weights.shape[1]].view(-1, 1, 1)).sum(dim=0).cpu().numpy()
        ax.imshow(
            h_map,
            cmap="viridis",
            alpha=0.7,
            extent=(0, img_large.shape[1], img_large.shape[0], 0),
            interpolation="nearest",
        )
    ax.set_title(f"INPUT (Weighted Image)\nClasse: {label}")
    ax.axis("off")


def plot_attention(ax, img_large, attn_grid):
    ax.imshow(img_large)
    ax.imshow(
        attn_grid,
        cmap="viridis",
        alpha=0.7,
        extent=(0, img_large.shape[1], img_large.shape[0], 0),
        interpolation="nearest",
    )
    ax.set_title("\nSelf-Attention Heatmap (branch Large)")
    ax.axis("off")


def debug_full_diagnostic(x_small, x_large, weights, model, alphas, labels, id_to_label, save_fp):

    # transform data to img arrays
    attn_grid = get_attention_grid(model, x_small, x_large, weights, alphas)
    img_large = tensor_to_img(x_large[0])
    if len(x_small[0].shape) == 3:
        img_small = tensor_to_img(x_small[0])
    else:
        img_small = [tensor_to_img(x_small[0][i]) for i in range(len(x_small[0]))]

    # init figure
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(2, 4, width_ratios=[1.2, 1.2, 0.8, 0.8])

    # img_large input plot
    label = id_to_label[int(labels[0])]
    ax1 = fig.add_subplot(gs[:, 0])
    plot_input(ax1, img_large, weights[0], alphas, label)

    # attention plot
    ax2 = fig.add_subplot(gs[:, 1])
    plot_attention(ax2, img_large, attn_grid)

    # img_small input plot
    if len(x_small[0].shape) == 4:
        n_patches = min(len(img_small), 8)
        inner_gs = gs[:, 2:].subgridspec(4, 2)
        axs = [fig.add_subplot(inner_gs[i // 2, i % 2]) for i in range(n_patches)]
        plot_yolo_patches(axs, img_small, n_patches)
    else:
        ax3 = fig.add_subplot(gs[:, 2])
        plot_segmented_img(ax3, img_small)

    plt.tight_layout()
    plt.savefig(save_fp)
    plt.close()

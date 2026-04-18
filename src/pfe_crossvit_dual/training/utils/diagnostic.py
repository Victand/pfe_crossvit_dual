import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def debug_full_diagnostic(
    x_small,
    img_large,
    weights,
    model,
    alphas,
    labels,
    classes,
    epoch,
    save_dir="saved/images",
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        _ = model(x_small, img_large, weights=weights, alpha=alphas)
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
            (weights[0] * alphas[: weights.shape[1]].view(-1, 1, 1))
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

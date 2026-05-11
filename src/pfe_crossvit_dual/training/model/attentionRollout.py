# TODO Ajouter un sanity check : faire rentrer une image avec la partie "importante"
#       et obstrué pour savoir si le modele s'effondre

"""
Dans un Transformer, chaque couche produit une matrice d'attention qui décrit l'interaction entre les tokens locaux.
    L'attention est locale à la couche, une couche se base sur la couche précedente, et non par rapport à l'image d'origine.
    Comme l'information se mélange, à la fin on ne sait pas quelles parties de l'image d'origine a était importante pour le modèle.

Attention rollout : technique d'interprétabilité conçue pour les Transformers qui trace le flux d'informations depuis l'image d'entrée
    jusqu'à la décision finale
1. A chaque couche on aggrége les tetes d'attention (moyenne ou autre methode), on obtient une matrice A_l à chaque couche.
2. On ajoute une matrice identité à A_l pour modéliser les connexions résiduelles (skip) : Â_l = 0.5*A_l + 0.5*Id
3. On normalise les Â_l de telle sorte que la somme de leurs lignes vaille 1
4. Rollout(l) = Â_l*Â_(l-1)*...*Â_1 : multiplication récursive pour obtenir la matrice de rollout de la couche l,
    qui lie les tokens de la couche l aux patchs de l'image initiale

Pour l'interpretation, on extrait la ligne correspondant au cls token de la matrice finale. C'est à dire les valeurs d'importances pour chaque patch.
On transforme le vecteur en grille (15x15 si image 240 et patch size 16) puis fait une interpolation pour superposer ça à l'image d'origine -> Heatmap


"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def get_attention_maps(model: DualCrossVit):
    """
    Extracts attention maps from the given DualCrossVit model
    Returns a map of attention maps per stage : 'small', 'large', 'fusion'
    """
    maps = {
        "small": [],  # Small Branch
        "large": [],  # Branche Large
        "fusion": [],  # Cross-Attention
    }

    # model.blocks contains MultiScaleBlock
    for ms_block in model.blocks:
        if ms_block.blocks is not None:
            # Branch Small
            small_branch_maps = []
            # ms_block is a RecordingBlock
            for block in ms_block.blocks[0]:
                small_branch_maps.append(block.attn.last_attn_map)
            maps["small"].append(small_branch_maps)

            # Branch Large
            large_branch_maps = []
            for block in ms_block.blocks[1]:
                large_branch_maps.append(block.attn.last_attn_map)
            maps["large"].append(large_branch_maps)

        # Fusion (Cross-Attention)
        fusion_maps = []
        for fusion_block in ms_block.fusion:
            # fusion_block is a CrossAttentionBlock
            if isinstance(fusion_block, nn.Sequential):
                for b in fusion_block:
                    fusion_maps.append(b.attn.last_attn_map)
            else:
                fusion_maps.append(fusion_block.attn.last_attn_map)
        maps["fusion"].append(fusion_maps)

    return maps


def process_layer_for_rollout(attn_tensor):
    """
    Prepare the layers for the rollout given the attention heads.
    """
    device = attn_tensor.device

    if attn_tensor.dim() == 4:
        result = attn_tensor[0].mean(dim=0)
    elif attn_tensor.dim() == 3:
        result = attn_tensor.mean(dim=0)
    else:
        result = attn_tensor

    id_ = torch.eye(result.shape[-1], device=device)
    result = 0.5 * result + 0.5 * id_
    result = result / result.sum(dim=-1, keepdim=True)
    return result


def rollout(attn_matrix_layers):
    """
    Performs attention rollout on given layers of attention tensors, assumed as oredered from input to output.
    """
    n = attn_matrix_layers[0].shape[-1]
    device = attn_matrix_layers[0].device

    global_rollout = torch.eye(n, device=device)

    for A_hat in attn_matrix_layers:
        global_rollout = A_hat @ global_rollout

    return global_rollout


def branches_attention_rollout(maps):
    """
    Performs attention rollout on both branches small and large.
    """

    results = {}
    for branch in ["small", "large"]:
        flat_layers = [layer for stage in maps[branch] for layer in stage]

        processed_layers = [process_layer_for_rollout(l) for l in flat_layers]

        results[branch] = rollout(processed_layers)

    return results


def getInfluenceVectors(main_branch: str, influence_branch: str, rollouts, maps):
    """
    Returns a direct vector for the main branch which informs on the decision making of the main branch using self attention.
    Returns a indirect vecrtor, which informs on the influence of the influence_branch on the main branch through fusion.
    """
    device = rollouts[main_branch].device

    branch_indices = {"small": 0, "large": 1}
    if main_branch not in branch_indices or influence_branch not in branch_indices:
        raise ValueError("Les branches doivent être 'small' ou 'large'")
    main_idx = branch_indices[main_branch]

    # Direct Vector

    # Takes CLS line (index 0), excludes itself (column 0)
    matrix_main = rollouts[main_branch].squeeze()
    direct_vector = matrix_main[0, 1:].to(device)

    # Indirect Vector

    # We take the fusion of the last layer
    last_fusion_matrix = maps["fusion"][-1][main_idx]

    # [1, Heads, 1, 1+N] -> [1+N]
    fusion_attn = last_fusion_matrix.mean(dim=1).squeeze().to(device)
    # Take out cls token
    fusion_attn = fusion_attn[1:]

    matrix_influence = rollouts[influence_branch].squeeze()
    rollout_influence_branch = matrix_influence[1:, 1:].to(device)

    indirect_vector = fusion_attn @ rollout_influence_branch

    return direct_vector, indirect_vector


def vector_to_heatmap(
    vector,
    branch_img_size: tuple[int, int],
    patch_size: tuple[int, int],
    target_size=(240, 240),
):
    """
    Convertit un vecteur d'attention en heatmap 2D normalisée.
    Gère les grilles asymétriques et utilise F.interpolate.
    """
    if vector.dim() == 1:
        vector = vector.unsqueeze(0)  # [1, N]

    b, n = vector.shape
    device = vector.device

    h_grid = branch_img_size[0] // patch_size[0]
    w_grid = branch_img_size[1] // patch_size[1]

    if h_grid * w_grid != n:
        h_grid = w_grid = int(n**0.5)

    heatmap = vector.view(b, 1, h_grid, w_grid)

    h_min = heatmap.view(b, -1).min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    h_max = heatmap.view(b, -1).max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)

    denom = h_max - h_min
    heatmap = torch.where(denom > 1e-8, (heatmap - h_min) / denom, heatmap)

    heatmap = F.interpolate(heatmap, size=target_size, mode="bilinear", align_corners=False)
    if b == 1:
        return heatmap.squeeze().detach().cpu().numpy()
    return heatmap.detach().cpu().numpy()


def show_overlay(img_tensor, heatmaps: dict, std, mean, title="Dual CrossViT Analysis"):
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = img_np * np.array(std) + np.array(mean)
    img_np = np.clip(img_np, 0, 1)

    ncols = 1 + len(list(heatmaps.items()))
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(title, fontsize=16)

    axes[0].imshow(img_np)
    axes[0].set_title("Image Originale")
    axes[0].axis("off")

    for k, (text, map) in enumerate(heatmaps.items()):
        im = axes[k + 1].imshow(map, cmap="jet")
        axes[k + 1].set_title(f"{text}")
        axes[k + 1].axis("off")
        fig.colorbar(im, ax=axes[k + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def perform_rollout(model, show_individuals=False):
    """
    Performs rollout on given model and plot heatmaps.
    """
    # the targeted size is the small branch
    small_img_size, large_img_size = model.img_size
    small_patch_size, large_patch_size = model.patch_size
    target_size = (small_img_size, small_img_size)

    attention_maps = get_attention_maps(model)
    # Rollouts on both branches
    rollouts = branches_attention_rollout(attention_maps)

    direct_vector, indirect_vector = getInfluenceVectors("small", "large", rollouts, attention_maps)

    map_direct, map_indirect = (
        vector_to_heatmap(
            direct_vector,
            (small_img_size, small_img_size),
            (small_patch_size, small_patch_size),
            target_size,
        ),
        vector_to_heatmap(
            indirect_vector,
            (large_img_size, large_img_size),
            (large_patch_size, large_patch_size),
            target_size,
        ),
    )

    final_heatmap = 0.5 * map_direct + 0.5 * map_indirect

    if show_individuals:
        heatmaps = {
            "Influence directe": map_direct,
            "Influence indirecte (Cross Attention)": map_indirect,
            "Rollout global": final_heatmap,
        }
    else:
        heatmaps = {"Rollout global": final_heatmap}

    return heatmaps


def compute_iou(heatmap, segmented_image, img_size=240, threshold=0.5):
    """
    Calcule l'IoU entre une heatmap d'attention et un masque binaire.

    """

    img_tensor = TF.to_tensor(
        TF.center_crop(TF.resize(segmented_image, int(img_size * 1.14)), img_size)
    )
    target_mask = (torch.sum(img_tensor, dim=0) > 0.1).cpu().numpy().astype(bool)

    if torch.is_tensor(heatmap):
        heatmap = heatmap.detach().cpu().numpy()

    # Normalization
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap_norm = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap_norm = heatmap

    pred_mask = (heatmap_norm >= threshold).astype(bool)

    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()

    if union == 0:
        return 0.0

    return intersection / union


def to_2d_resized(vec, img_size, patch_size, target_size):
    b = vec.shape[0]
    vec = vec.reshape(b, -1)

    # 1. Normalisation Min-Max locale pour redonner du "poids" aux pixels
    # Cela évite que le pow(2) ne rende tout invisible
    v_min = vec.min(dim=1, keepdim=True)[0]
    v_max = vec.max(dim=1, keepdim=True)[0]
    vec = (vec - v_min) / (v_max - v_min + 1e-8)  # 1e-8 pour éviter div par zero

    h_grid, w_grid = img_size // patch_size, img_size // patch_size
    if h_grid * w_grid != vec.shape[1]:
        side = int(vec.shape[1] ** 0.5)
        h_grid = w_grid = side

    grid = vec.view(b, 1, h_grid, w_grid)

    # 2. Maintenant le pow(2) va vraiment séparer le "grain de l'ivraie"
    # Les valeurs fortes (proches de 1) restent fortes, les faibles s'écrasent.
    grid = torch.pow(grid, 2)

    return F.interpolate(grid, size=target_size, mode="bilinear", align_corners=False)


def get_trainable_heatmap(model, device):
    """
    Returns an heatmap of attention rollout that can derivated and used in training.
    """
    maps = get_attention_maps(model)
    results = {}

    # Référence pour l'espace commun (Branche Small)
    ref_size = (model.img_size[0], model.img_size[0])

    for branch in ["small", "large"]:
        layers = [l for stage in maps[branch] for l in stage]
        n = layers[0].shape[-1]

        res = torch.eye(n, device=device).unsqueeze(0)
        identity = torch.eye(n, device=device)

        for A in layers:
            A = A.to(device)
            A_hat = 0.5 * (A.mean(dim=1) + identity)
            A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True)
            res = A_hat @ res
        results[branch] = res

    # Extraction fusion et alignement
    fusion_attn = maps["fusion"][-1][0].to(device).mean(dim=1).squeeze(1)
    rollout_large = results["large"]

    n_tokens_rollout = rollout_large.shape[-1]
    n_tokens_fusion = fusion_attn.shape[-1]

    if n_tokens_fusion > n_tokens_rollout:
        fusion_attn = fusion_attn[:, -n_tokens_rollout:]
    elif n_tokens_rollout > n_tokens_fusion:
        rollout_large = rollout_large[:, -n_tokens_fusion:, -n_tokens_fusion:]

    # Vecteurs bruts (on s'assure qu'ils sont en 3D avant multiplication si besoin)
    if fusion_attn.dim() == 2:
        fusion_attn = fusion_attn.unsqueeze(1)

    indirect_vec_raw = (fusion_attn @ rollout_large)[:, :, 1:]
    direct_vec_raw = results["small"][:, 0, 1:]

    # Projection spatiale
    map_direct = to_2d_resized(
        direct_vec_raw,
        img_size=model.img_size[0],
        patch_size=model.patch_size[0],
        target_size=ref_size,
    )

    map_indirect = to_2d_resized(
        indirect_vec_raw,
        img_size=model.img_size[1],
        patch_size=model.patch_size[1],
        target_size=ref_size,
    )

    return 0.5 * (map_direct + map_indirect)


def get_iou_training(model, heatmap, segmented_image):
    """
    Get Iou function for train loop.
    Normalizes heatmap, creates binary mask, target mask based on input image and measures IoU.
    """
    # Normalization with min-max
    b = heatmap.shape[0]
    h_flat = heatmap.view(b, -1)
    h_min = h_flat.min(dim=1, keepdim=True)[0]
    h_max = h_flat.max(dim=1, keepdim=True)[0]
    heatmap_norm = (h_flat - h_min) / (h_max - h_min + 1e-8)
    heatmap_norm = heatmap_norm.view_as(heatmap)

    # Threshold based on attention (heatmap)
    predicted_mask = (heatmap_norm > 0.5).float()

    # Ground truth
    target_mask = (torch.abs(segmented_image).sum(dim=1, keepdim=True) > 0.1).float()
    target_mask = F.interpolate(
        target_mask, size=(model.img_size[0], model.img_size[0]), mode="nearest"
    )

    intersection = (predicted_mask * target_mask).sum(dim=(1, 2, 3))
    union = (predicted_mask + target_mask).clamp(0, 1).sum(dim=(1, 2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean().item()

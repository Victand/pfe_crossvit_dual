from pathlib import Path
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnTF
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from PIL import Image, ImageFile
from pfe_crossvit_dual.training.utils.weight_functions import linear_
from pfe_crossvit_dual.constants.paths import DATA_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True


ALL_TRANSFORMS = ["random_crop", "hflip", "color_jitter", "random_erase"]


class DualInputDataset(Dataset):
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        is_train: bool = True,
        img_size=(240, 240),
        patch_size=(16, 16),
        classes=("class1", "class2"),
        paths=("original", "segmented"),
        weighed_patches=False,
        weight_function=linear_,
        use_yolo_weights=False,
        num_patches=16,
        patch_quotas={0:2, 1:0, 2:12, 3:2},
        precompute=False
    ):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.image_size = img_size
        self.patch_size = patch_size
        self.classes = classes
        self.paths = paths
        self.weighed_patches = weighed_patches
        self.weight_function = weight_function
        self.numbranches = 2
        self.use_yolo_weights = use_yolo_weights
        self.num_patches = num_patches
        self.patch_quotas = patch_quotas
        self.precompute = precompute

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.all_transforms = ALL_TRANSFORMS
        self.active_transforms = {transform: True for transform in self.all_transforms}
        self.samples = []
        self.cache = {}
        self.classes_count = {self.classes[0]: 0, self.classes[1]: 0}
        self.load_samples()

        if self.precompute:
            self.data = []
            for idx in tqdm(range(len(self.samples)), desc="precomputing dataset"):
                self.data.append(self[idx])

    def load_samples(self):
        phase = "train" if self.is_train else "val"
        for label_int, _class in enumerate(self.classes):
            original_path = self.data_dir / phase / self.paths[0] / _class
            segmented_path = self.data_dir / phase / self.paths[1] / _class
            if not (original_path.exists() and segmented_path.exists()):
                continue
            for original_img_p in original_path.iterdir():
                image = original_img_p.name
                segmented_img_p = segmented_path / image
                if not segmented_img_p.exists():
                    continue
                if self.use_yolo_weights:
                    weight_p = (
                        original_img_p.parent / f"{original_img_p.stem}_weights.pt"
                    )
                    if not weight_p.exists():
                        continue
                self.samples.append((original_img_p, segmented_img_p, label_int))
                self.classes_count[self.classes[label_int]] += 1
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.precompute:
            return self.data[idx]
        
        original_image_p, segmented_image_p, label_int = self.samples[idx]
        original_image = Image.open(original_image_p).convert("RGB")

        # --- CAS CLASSIQUE ---
        if not self.use_yolo_weights:
            segmented_image = Image.open(segmented_image_p).convert("RGB")
            img_large, img_seg = self.synchronized_transform(
                original_image, segmented_image, is_heatmap=False
            )
            img_large, img_seg = normalize_and_erase(self, img_large, img_seg)
            weights = (
                self.patches_weights(img_seg, self.patch_size[0], self.weight_function)
                if self.weighed_patches
                else torch.empty(0)
            )
            return img_seg, img_large, label_int, weights

        # --- CAS YOLO ---
        # 1. Charger la heatmap brute d'abord
        weight_p = original_image_p.parent / f"{original_image_p.stem}_weights.pt"
        data = torch.load(weight_p)
        weights_raw = data["global_heatmap"].squeeze(0)

        # 2. Appliquer les mêmes transformations à l'image et à la heatmap en même temps
        img_large, weights = self.synchronized_transform(
            original_image, weights_raw, is_heatmap=True
        )

        # 3. Normaliser l'image (les poids n'en ont pas besoin)
        img_large = TF.normalize(img_large, self.mean, self.std)

        # 4. Gérer les patchs (Le reste de ton code ne change pas)
        all_patches_data = data.get("patches", [])
        final_list = []
        if all_patches_data:
            # Classement par ID direct (puisque tes .pt sont maintenant corrects)
            by_class = {i: [] for i in range(10)}
            for p in all_patches_data:
                by_class[p["class_id"]].append(p["tensor"])

            # Remplissage par quotas
            for class_id, quota in self.patch_quotas.items():
                candidates = by_class.get(class_id, [])
                random.shuffle(candidates)
                final_list.extend(candidates[:quota])

            # Remplissage si l'image manque de tiges
            if len(final_list) < self.num_patches:
                used_ids = [id(t) for t in final_list]
                remains = [
                    p["tensor"]
                    for p in all_patches_data
                    if id(p["tensor"]) not in used_ids
                ]
                random.shuffle(remains)
                final_list.extend(remains[: (self.num_patches - len(final_list))])

        # Ajustement final
        if len(final_list) > self.num_patches:
            final_list = final_list[: self.num_patches]
        elif len(final_list) < self.num_patches:
            padding = [
                torch.zeros((3, 224, 224))
                for _ in range(self.num_patches - len(final_list))
            ]
            final_list.extend(padding)

        selected_patches = [
            TF.normalize(p, self.mean, self.std) if p.max() > 0 else p
            for p in final_list
        ]
        return torch.stack(selected_patches), img_large, label_int, weights

    def set_active_transforms(self, active_transforms_list):
        self.active_transforms = {transform: False for transform in self.all_transforms}
        for tf in active_transforms_list:
            if tf in self.all_transforms:
                self.active_transforms[tf] = True

    def patches_weights(self, segmented_tensor, patch_size: int, f=lambda x: x + 1e-7):
        mask = (
            (torch.sum(segmented_tensor, dim=0) > 0)
            .float()
            .unsqueeze(dim=0)
            .unsqueeze(0)
        )
        patches = nnTF.unfold(mask, kernel_size=patch_size, stride=patch_size).squeeze(
            0
        )
        ratios = torch.mean(patches, dim=0)
        num_p = ratios.numel()
        w_norm = (
            torch.ones(num_p)
            if not ratios.sum() > 0
            else f(ratios) * num_p / (f(ratios).sum() + 1e-8)
        )
        return torch.unsqueeze(w_norm, dim=1)

    def synchronized_transform(self, img, mask_or_heatmap, is_heatmap=False):
        # 1. Calcul de la nouvelle taille
        new_size = (int(self.image_size[0] * 1.14), int(self.image_size[1] * 1.14))

        # 2. Conversion de l'image principale en Tensor
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)

        # 3. Redimensionnement
        img = TF.resize(img, new_size, antialias=True)

        if isinstance(mask_or_heatmap, torch.Tensor):
            # Cas YOLO (Heatmap) : on lisse les valeurs avec Bilinear
            interp = (
                TF.InterpolationMode.BILINEAR
                if is_heatmap
                else TF.InterpolationMode.NEAREST
            )
            mask_or_heatmap = TF.resize(
                mask_or_heatmap, new_size, interpolation=interp, antialias=True
            )
        else:
            # Cas normal (Image PIL segmentée)
            mask_or_heatmap = TF.to_tensor(mask_or_heatmap)
            mask_or_heatmap = TF.resize(
                mask_or_heatmap,
                new_size,
                interpolation=TF.InterpolationMode.NEAREST,
                antialias=True,
            )

        # 4. Transformations Spatiales
        if self.is_train:
            # Coupe aléatoire (Crop)
            if self.active_transforms["random_crop"]:
                i, j, h, w = transforms.RandomCrop.get_params(
                    img, output_size=self.image_size
                )
                img = TF.crop(img, i, j, h, w)
                mask_or_heatmap = TF.crop(mask_or_heatmap, i, j, h, w)

            # Effet miroir (Flip)
            if self.active_transforms["hflip"] and random.random() > 0.5:
                img = TF.hflip(img)
                mask_or_heatmap = TF.hflip(mask_or_heatmap)

            # 5. Changement de couleurs (UNIQUEMENT SUR L'IMAGE)
            if self.active_transforms["color_jitter"]:
                p = transforms.ColorJitter.get_params(
                    [0.6, 1.4], [0.6, 1.4], [0.6, 1.4], [-0.1, 0.1]
                )
                for fn_id in p[0]:
                    if fn_id == 0:
                        img = TF.adjust_brightness(img, p[1])
                    elif fn_id == 1:
                        img = TF.adjust_contrast(img, p[2])
                    elif fn_id == 2:
                        img = TF.adjust_saturation(img, p[3])
                    elif fn_id == 3:
                        img = TF.adjust_hue(img, p[4])

        # 6. Recadrage central si on est en validation ou sans random_crop
        if (not self.active_transforms["random_crop"]) or (not self.is_train):
            img = TF.center_crop(img, self.image_size)
            mask_or_heatmap = TF.center_crop(mask_or_heatmap, self.image_size)

        return img, mask_or_heatmap


def normalize_and_erase(dualdataset, img1, img2):
    img1, img2 = (
        TF.normalize(img1, dualdataset.mean, dualdataset.std),
        TF.normalize(img2, dualdataset.mean, dualdataset.std),
    )
    if (
        dualdataset.is_train
        and dualdataset.active_transforms["random_erase"]
        and random.random() < 0.1
    ):
        i, j, h, w, v = transforms.RandomErasing.get_params(
            img1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=None
        )
        img1, img2 = TF.erase(img1, i, j, h, w, v), TF.erase(img2, i, j, h, w, v)
    return img1, img2


def prepare_dataloaders(train_ds, val_ds, batch_size, num_workers):
    t_ld = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    v_ld = DataLoader(
        val_ds, batch_size=int(2 * batch_size), num_workers=num_workers, shuffle=False
    )
    return t_ld, v_ld


import matplotlib.colors as colors


def plot_samples_with_weights(samples, classes, paths, alpha_config, suptitle=""):
    orig_p, _, label_int = samples[0]
    weight_p = orig_p.parent / f"{orig_p.stem}_weights.pt"

    if not weight_p.exists():
        return

    data = torch.load(weight_p)
    w_multi = data["global_heatmap"]
    patches = data.get("patches", [])

    if not isinstance(alpha_config, torch.Tensor):
        alpha_config = torch.tensor(alpha_config)

    with torch.no_grad():
        w_mixed = (w_multi * alpha_config.view(1, -1, 1, 1)).sum(dim=1).squeeze()

    w_mixed_np = np.rot90(w_mixed.numpy(), k=1)
    import matplotlib

    matplotlib.rcParams["toolbar"] = "None"  # Désactive la barre d'outils qui crash
    plt.close("all")
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])

    ax_main = fig.add_subplot(gs[0, :])
    img_orig = Image.open(orig_p).transpose(Image.ROTATE_90)

    ax_main.imshow(img_orig)
    im = ax_main.imshow(
        w_mixed_np,
        cmap="jet",
        alpha=0.5,
        extent=[0, img_orig.size[0], img_orig.size[1], 0],
        interpolation="nearest",
        norm=colors.LogNorm(vmin=0.05, vmax=15),
    )

    cbar = plt.colorbar(im, ax=ax_main)
    cbar.set_ticks([0.1, 1.0, 10.0])
    cbar.set_ticklabels(["0.1 (Fond)", "1.0 (Feuille)", "10.0 (Tige)"])

    ax_main.set_title(
        f"Visualisation Guidage REEL | Classe: {classes[label_int]}\n{suptitle}"
    )

    if len(patches) > 0:
        stems = [p for p in patches if p["class_id"] == 2]
        others = [p for p in patches if p["class_id"] != 2]
        selected_patches = (stems + others)[:3]

        for idx, p_data in enumerate(selected_patches):
            ax_p = fig.add_subplot(gs[1, idx])
            p_img = p_data["tensor"].permute(1, 2, 0).numpy()
            p_img_plot = (p_img - p_img.min()) / (p_img.max() - p_img.min() + 1e-8)
            ax_p.imshow(p_img_plot)
            ax_p.set_title(f"Patch Small | ID: {p_data['class_id']}")
            ax_p.axis("off")

    plt.tight_layout()
    plt.show()

    # --- B) Les vignettes : Patches Small sélectionnés ---
    if len(patches) > 0:
        # On filtre pour mettre les tiges (ID 2) en premier (puisque c'est ta priorité)
        stems = [p for p in patches if p["class_id"] == 2]
        others = [p for p in patches if p["class_id"] != 2]

        # On prend 3 patches (les tiges d'abord, puis le reste pour boucher les trous)
        selected_patches = (stems + others)[:3]

        for idx, p_data in enumerate(selected_patches):
            ax_p = fig.add_subplot(
                gs[1, idx]
            )  # Occupe une colonne de la deuxième ligne
            # Conversion (C, H, W) -> (H, W, C) pour matplotlib
            p_img = p_data["tensor"].permute(1, 2, 0).numpy()

            # Normalisation simple pour l'affichage des vignettes
            p_img_plot = (p_img - p_img.min()) / (p_img.max() - p_img.min() + 1e-8)
            ax_p.imshow(p_img_plot)
            ax_p.set_title(f"Patch Small | Class ID: {p_data['class_id']}")
            ax_p.axis("off")  # Cache les axes pour que ce soit plus propre

    plt.tight_layout()
    plt.show()


def find_two_samples(samples):
    s1 = random.choice(samples)
    s2 = random.choice([s for s in samples if s[2] != s1[2]])
    return [s1, s2]

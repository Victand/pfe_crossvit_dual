from pathlib import Path
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as tf
from torchvision.tv_tensors import Image
import torch
import torch.nn.functional as nnTF
from collections import defaultdict
from tqdm import tqdm
from typing import Literal

from pfe_crossvit_dual.training.utils.weight_functions import linear_
from pfe_crossvit_dual.constants.paths import CACHE_DIR
from pfe_crossvit_dual.training.utils.io import read_image
from pfe_crossvit_dual.training.utils.transforms import (
    apply_colorjitter,
    scale_translation,
    scale_crop_params,
)


IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


SBranchType = Literal["original", "segmented", "yolo_patches"]
LBranchType = Literal["original", "segmented", "yolo_patches"]
LBranchWeightType = Literal["ratio", "yolo_masks"] | None

TransformType = Literal["random_crop", "random_hflip", "random_affine", "color_jitter", "random_grayscale", "gaussian_blur", "random_erasing"]


class DualInputDataset(Dataset):
    def __init__(
        self,
        img_paths: list[Path],
        label_to_id: dict[str, int],
        is_train: bool = True,
        img_size=(240, 224),
        branch_small: SBranchType = "segmented",
        branch_large: LBranchType = "original",
        branch_large_weight: LBranchWeightType = "ratio",
        ratio_patch_size=32,
        ratio_weight_function=linear_,
        transforms: list[TransformType] =[],
        precompute=False,
        use_cache=False,
        store_cache=False,
        yolo_patch_count=16,
        yolo_patch_quotas={0: 2, 1: 0, 2: 12, 3: 2},
    ):
        # data
        self.label_to_id = label_to_id
        self.cache_dir = Path(CACHE_DIR)
        self.is_train = is_train
        self.img_size_small = [img_size[0], img_size[0]]
        self.img_size_large = [img_size[1], img_size[1]]
        self.ratio_patch_size = ratio_patch_size
        self.branch_small = branch_small
        self.branch_large = branch_large
        self.branch_large_weight = branch_large_weight
        self.transforms = transforms

        self.need_original = self.branch_large == "original" or self.branch_small == "original"
        self.need_segmented = (
            self.branch_large == "segmented"
            or self.branch_small == "segmented"
            or self.branch_large_weight == "ratio"
        )
        self.need_yolo_weight = self.branch_large_weight == "yolo_masks"
        self.need_yolo_patches = self.branch_small == "yolo_patches"

        # ratio weights
        self.weight_function = ratio_weight_function
        # yolo weights
        self.num_patches = yolo_patch_count
        self.patch_quotas = yolo_patch_quotas
        # samples
        self.samples = {}
        self._index_files(img_paths)
        # precomputing
        self.precomputed = precompute
        if precompute:
            self._precompute_all(use_cache, store_cache)

    def _index_files(self, original_img_paths: list[Path]):
        for p in original_img_paths:
            original_img = p
            segmented_img = p.parent.parent.parent / "segmented" / p.parent.name / p.name
            yolo_data = p.with_name(f"{p.stem}_weights.pt")
            label = self.label_to_id[str(p.parent.name)]

            self.samples[original_img.stem] = {
                "original": original_img,
                "segmented": segmented_img,
                "yolo_data": yolo_data,
                "label": label,
            }

    def _load_cache(self, cache_fp):
        try:
            cache = torch.load(cache_fp, weights_only=False)
            print(f"successfully loaded cache {cache_fp.parent.name} {cache_fp.name}.")
            return cache
        except Exception:
            print(f"failed loading cache {cache_fp.parent.name} {cache_fp.name}.")
            return {}

    def _precompute_all(self, use_cache=True, store_cache=True):
        """precompute all io tasks and expensive deterministic tasks (ie not random transforms)"""
        dataset = list(self.samples.values())[0]["original"].parent.parent.parent.name
        phase = "train" if self.is_train else "val"
        cache_subdir = self.cache_dir / f"precomputed_{dataset}_{phase}"

        keys = ["label", "original", "segmented", "yolo_patches", "yolo_weight"]
        data_needed = [k for k in keys if getattr(self, f"need_{k}", True)]
        data = {k: {} for k in data_needed}

        # get cache
        if use_cache:
            for k in data_needed.copy():
                cache_fp = cache_subdir / f"{k}.pt"
                data[k] = self._load_cache(cache_fp)
                if data[k]:
                    data_needed.remove(k)

        # compute not cached data
        if data_needed:
            print(f"precomputing {data_needed}")
            for img_id in tqdm(self.samples, desc="precomputing data"):
                sample_data = self._get_sample_data(img_id, keys=data_needed, resize=True)
                if sample_data is None:
                    continue
                for k, v in sample_data.items():
                    data[k][img_id] = v

        # assemble
        common_keys = set(data["label"].keys())
        for d in data.values():
            common_keys &= set(d.keys())

        self._cache = [{k: d[i] for k, d in data.items()} for i in common_keys]

        # save cache
        if store_cache:
            cache_subdir.mkdir(exist_ok=True)
            for k in data_needed:
                torch.save(data[k], cache_subdir / f"{k}.pt")

    def __getitem__(self, idx):
        if self.precomputed:
            sample_data = self._cache[idx]
        else:
            sample_data = self._get_sample_data(idx)
            if sample_data is None:
                return None, None, None, None

        # x_large
        x_large = sample_data[self.branch_large]

        # x_small
        x_small = sample_data[self.branch_small]

        # weight
        if self.branch_large_weight == "yolo_masks":
            weight = sample_data["yolo_weight"]
        elif self.branch_large_weight == "ratio":
            weight = self._patches_weights(
                sample_data["segmented"], self.ratio_patch_size, self.weight_function
            )
        else:
            weight = torch.empty(0)  # TODO test

        # transform
        x_small, x_large, weight = self._joint_transform(x_small, x_large, weight)

        # stack patches
        if isinstance(x_small, list):
            x_small = torch.stack(x_small)

        return x_small, x_large, weight, sample_data["label"]

    def __len__(self):
        return len(self._cache) if self.precomputed else len(self.samples)

    def _joint_transform(self, x_small, x_large, weight, mean=IMGNET_MEAN, std=IMGNET_STD):
        """Apply transformations to all images at the same time"""

        def apply_to_small(fn):
            if isinstance(x_small, list):
                return [fn(x) for x in x_small]
            else:
                return fn(x_small)

        if "random_crop" in self.transforms and self.is_train:
            params_l = tf.RandomResizedCrop.get_params(
                x_large, scale=[0.85, 1.0], ratio=[0.75, 1.33]
            )
            params_s = scale_crop_params(*params_l, x_large.shape[-2:], self.img_size_small)
            params_w = scale_crop_params(*params_l, x_large.shape[-2:], weight.shape[-2:])
            x_large = F.resized_crop(x_large, *params_l, self.img_size_large)
            weight = F.resized_crop(
                weight, *params_w, weight.shape[-2:], tf.InterpolationMode.BILINEAR
            )
            if not isinstance(x_small, list): # don't apply to yolo patches
                x_small = apply_to_small(lambda x: F.resized_crop(x, *params_s, self.img_size_small))

        if "random_hflip" in self.transforms and self.is_train:
            if random.random() < 0.5:
                x_large = F.hflip(x_large)
                weight = F.hflip(weight)
                x_small = apply_to_small(lambda x: F.hflip(x))

        if "random_affine" in self.transforms and self.is_train:
            angle, translations_l, scale, shear = tf.RandomAffine.get_params(
                [-15, 15], [0.1, 0.1], [0.9, 1.1], [-5, 5], self.img_size_large
            )
            translations_s = scale_translation(
                translations_l, self.img_size_large, self.img_size_small
            )
            translations_w = scale_translation(
                translations_l, self.img_size_large, weight.shape[-2:]
            )
            x_large = F.affine(x_large, angle, translations_l, scale, shear)  # type: ignore
            if not isinstance(x_small, list): # dont apply to yolo patches
                x_small = apply_to_small(lambda x: F.affine(x, angle, translations_s, scale, shear))  # type: ignore
            weight = F.affine(
                weight,
                angle,
                translations_w,  # type: ignore
                scale,
                shear,  # type: ignore
                interpolation=tf.InterpolationMode.BILINEAR,
            )

        if "color_jitter" in self.transforms and self.is_train:
            params = tf.ColorJitter.get_params([0.6, 1.4], [0.6, 1.4], [0.6, 1.4], [-0.1, 0.1])
            x_large = apply_colorjitter(x_large, *params)
            x_small = apply_to_small(lambda x: apply_colorjitter(x, *params))

        if "random_grayscale" in self.transforms and self.is_train:
            if random.random() < 0.2:
                x_large = F.rgb_to_grayscale(x_large, num_output_channels=3)
                x_small = apply_to_small(lambda x: F.rgb_to_grayscale(x, num_output_channels=3))

        if "gaussian_blur" in self.transforms and self.is_train:
            sigma = tf.GaussianBlur.get_params(0.1, 2)
            x_large = F.gaussian_blur(x_large, 3, sigma)  # type: ignore
            x_small = apply_to_small(lambda x: F.gaussian_blur(x, 3, sigma))  # type: ignore

        if "random_erasing" in self.transforms and self.is_train:
            re = tf.RandomErasing(0.3)
            x_large = re(x_large)
            x_small = apply_to_small(lambda x: re(x))

        x_large = F.resize(x_large, self.img_size_large)
        x_small = apply_to_small(lambda x: F.resize(x, self.img_size_small))
        x_large = F.normalize(x_large, mean, std)
        x_small = apply_to_small(lambda x: F.normalize(x, mean, std))

        return x_small, x_large, weight

    def _patches_weights(self, segmented_tensor, patch_size: int, f=linear_):
        mask = (torch.sum(segmented_tensor, dim=0) > 0).float().unsqueeze(dim=0)
        patches = nnTF.unfold(mask, kernel_size=patch_size, stride=patch_size)
        ratios = torch.mean(patches, dim=0)
        num_p = ratios.numel()
        w_norm = (
            torch.ones(num_p)
            if not ratios.sum() > 0
            else f(ratios) * num_p / (f(ratios).sum() + 1e-8)
        )
        n_patch = mask.shape[-1] // patch_size
        w_norm = w_norm.view(n_patch, n_patch)
        return w_norm.unsqueeze(0)  # torch.unsqueeze(w_norm, dim=1)

    def _select_patches(self, yolo_patches):
        patches = []

        # get patches by class
        by_class = defaultdict(list)
        for p in yolo_patches:
            by_class[p["class_id"]].append(Image(p["tensor"]))

        # fill quotas
        for class_id, quota in self.patch_quotas.items():
            candidates = by_class.get(class_id, [])
            random.shuffle(candidates)
            patches.extend(candidates[:quota])

        # fill if missing patches
        if len(patches) < self.num_patches:
            used_ids = [id(t) for t in patches]
            remains = [Image(p["tensor"]) for p in yolo_patches if id(p["tensor"]) not in used_ids]
            random.shuffle(remains)
            patches.extend(remains[: (self.num_patches - len(patches))])

        # ensure correct number of patches
        patches = patches[: self.num_patches]
        if len(patches) < self.num_patches:
            padding = [
                torch.zeros((3, *self.img_size_small))
                for _ in range(self.num_patches - len(patches))
            ]
            patches.extend(padding)

        return patches

    def _get_sample_data(
        self,
        idx,
        keys=["original", "segmented", "yolo_patches", "yolo_weight", "label"],
        resize=False,
    ):
        try:
            ret = {}
            for k in keys:
                fname = "yolo_data" if "yolo" in k else k
                fp = self.samples[idx][fname]
                if k == "label":
                    ret[k] = self.samples[idx][k]
                if k == "original":
                    original = read_image(fp)
                    if resize:
                        original = F.resize(original, self.img_size_large)
                    ret[k] = original
                if k == "segmented":
                    segmented = read_image(fp)
                    if resize:
                        segmented = F.resize(segmented, self.img_size_small)
                    ret[k] = segmented
                yolo_data = None
                if k == "yolo_patches":
                    yolo_data = torch.load(fp)
                    ret[k] = self._select_patches(yolo_data["patches"])
                if k == "yolo_weight":
                    yolo_data = torch.load(fp) if yolo_data is None else yolo_data
                    ret[k] = yolo_data["global_heatmap"].squeeze(0)

            return ret

        except Exception as e:
            print("error processing sampling")
            print(f"error: {e}")
            return None

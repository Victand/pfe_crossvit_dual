from pathlib import Path
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import Image
import torchvision.io as io
import torch
import torch.nn.functional as nnTF
import PIL
from PIL import ImageFile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from pfe_crossvit_dual.training.utils.weight_functions import linear_
from pfe_crossvit_dual.constants.paths import CACHE_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True


IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class DualInputDataset(Dataset):
    def __init__(
        self,
        img_paths: list[Path],
        label_to_id: dict[str, int],
        is_train: bool = True,
        img_size=(240, 240),
        patch_size=(16, 16),
        weighted_patches=False,
        use_yolo_weights=False,
        weight_function=linear_,
        transforms=None,
        precompute=False,
        store_cache=False,
        precompute_workers=4,
        num_patches=16,
        patch_quotas={0: 2, 1: 0, 2: 12, 3: 2},
    ):
        # data
        self.label_to_id = label_to_id
        self.cache_dir = Path(CACHE_DIR)
        self.is_train = is_train
        self.image_size = img_size
        self.patch_size = patch_size

        # ratio weights
        self.weighted_patches = weighted_patches
        self.weight_function = weight_function
        # yolo weights
        self.use_yolo_weights = use_yolo_weights
        self.num_patches = num_patches
        self.patch_quotas = patch_quotas
        # transforms
        self._build_transforms(transforms if transforms is not None else [])
        # samples
        self.samples = []
        self._index_files(img_paths)
        # precomputing
        self.precomputed = precompute
        if precompute:
            self._precompute_all(store_cache, precompute_workers)

    def _index_files(self, original_img_paths: list[Path]):
        for p in original_img_paths:
            original_img = p
            segmented_img = (
                p.parent.parent.parent / "segmented" / p.parent.name / p.name
            )
            weight = (
                p.with_name(f"{p.stem}_weights.pt") if self.use_yolo_weights else None
            )
            label = self.label_to_id[str(p.parent.name)]

            self.samples.append((original_img, segmented_img, weight, label))

    def _build_transforms(self, transforms, mean=IMGNET_MEAN, std=IMGNET_STD):
        # random erase
        self.random_erase = "random_erase" in transforms

        # sptatial transforms
        spatial_tf = []
        if "random_crop" in transforms:
            spatial_tf.append(v2.RandomResizedCrop(self.image_size, scale=(0.85, 1.0)))
        if "random_hflip" in transforms:
            spatial_tf.append(v2.RandomHorizontalFlip())
        self.spatial_tf = v2.Compose(spatial_tf) if spatial_tf else v2.Identity()
        if "random_affine" in transforms:
            spatial_tf.append(
                v2.RandomAffine(
                    degrees=10,  # type: ignore
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    shear=5,
                )
            )

        # color transforms
        color_tf = []
        if "color_jitter" in transforms:
            color_tf.append(
                v2.ColorJitter([0.6, 1.4], [0.6, 1.4], [0.6, 1.4], [-0.1, 0.1])
            )
        if "random_grayscale" in transforms:
            color_tf.append(v2.RandomGrayscale(p=0.1))
        if "gaussian_blur" in transforms:
            color_tf.append(v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
        self.color_tf = v2.Compose(color_tf) if color_tf else v2.Identity()

        # normalization
        self.normalize = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    def _precompute_all(self, store_cache=True, num_workers=None):
        """precompute all io tasks and expensive deterministic tasks (ie not random transforms)"""
        dataset = self.samples[0][0].parent.parent.parent.name
        phase = "train" if self.is_train else "val"
        datatype = "yolo" if self.use_yolo_weights else "ratio"
        cache_fp = self.cache_dir / f"precomputed_{dataset}_{datatype}_{phase}.pt"

        if store_cache and cache_fp.is_file():
            self._cache = torch.load(cache_fp)
            if len(self._cache) == len(self.samples):
                print(f"using cached precomputed data at path {cache_fp}")
                return
            else:
                print("cached data invalid (different lengths), continuing...")

        self._cache = []

        tasks = [
            (
                img_p,
                seg_p,
                weight_p,
                label,
                self.image_size,
                self.use_yolo_weights,
                self._select_patches,
            )
            for img_p, seg_p, weight_p, label in self.samples
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_sample, t) for t in tasks]

            for f in tqdm(
                as_completed(futures), total=len(futures), desc="precomputing dataset"
            ):
                self._cache.append(f.result())

        if store_cache:
            torch.save(self._cache, cache_fp)

    def _getitem_ratio_weight(self, idx):
        """
        Get items with ratio weights
        """
        # get data
        if self.precomputed:
            img, img_seg, label = self._cache[idx]
        else:
            image_p, seg_image_p, _, label = self.samples[idx]
            img = _read_image(image_p)
            img_seg = _read_image(seg_image_p)

        # transform
        img = Image(img)
        img_seg = Image(img_seg)
        if self.is_train:
            img, img_seg = self.spatial_tf(img, img_seg)
            img = self.color_tf(img)
            if self.random_erase:
                img, img_seg = _erase_pair(img, img_seg)
        else:
            img = TF.resize(img, self.image_size)
            img_seg = TF.resize(img_seg, self.image_size)
        img, img_seg = self.normalize(img, img_seg)

        # get weight
        weights = (
            self._patches_weights(img_seg, self.patch_size[0], self.weight_function)
            if self.weighted_patches
            else torch.empty(0)
        )

        return img_seg, img, weights, label

    def _getitem_yolo_weight(self, idx):
        """
        Get items with yolo weights
        """
        # get data
        if self.precomputed:
            img, mask, patches, label = self._cache[idx]
        else:
            image_p, _, weight_p, label = self.samples[idx]

            img = _read_image(image_p)
            yolo_weight = torch.load(weight_p)
            mask = yolo_weight["global_heatmap"].squeeze(0)
            # get patches
            yolo_patches = yolo_weight["patches"]
            patches = self._select_patches(yolo_patches)

        # transforms
        img = Image(img)
        if self.is_train:
            img = self.spatial_tf(img)
            patches = torch.stack([self.spatial_tf(p) for p in patches])
            img = self.color_tf(img)
            patches = torch.stack([self.color_tf(p) for p in patches])
            if self.random_erase:
                img = _erase(img)
                patches = torch.stack([_erase(p) for p in patches])
        else:
            img = TF.resize(img, self.image_size)
            # patches resize to 224 224 already handled in preprocessing
        img = self.normalize(img)
        patches = torch.stack([self.normalize(p) for p in patches])

        return patches, img, mask, label

    def __getitem__(self, idx):
        if not self.use_yolo_weights:
            return self._getitem_ratio_weight(idx)
        else:
            return self._getitem_yolo_weight(idx)

    def __len__(self):
        return len(self.samples)

    def _patches_weights(self, segmented_tensor, patch_size: int, f=linear_):
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
            remains = [
                Image(p["tensor"])
                for p in yolo_patches
                if id(p["tensor"]) not in used_ids
            ]
            random.shuffle(remains)
            patches.extend(remains[: (self.num_patches - len(patches))])

            # ensure correct number of patches
        patches = patches[: self.num_patches]
        if len(patches) < self.num_patches:
            padding = [
                torch.zeros((3, 224, 224))
                for _ in range(self.num_patches - len(patches))
            ]
            patches.extend(padding)

        return torch.stack(patches)


def _process_sample(args):
    img_p, seg_p, weight_p, label, image_size, use_yolo_weights, select_patches = args

    img = _read_image(img_p)
    img = Image(img)
    img = TF.resize(img, image_size)

    if not use_yolo_weights:
        seg = _read_image(seg_p)
        seg = Image(seg)
        seg = TF.resize(seg, image_size)
        return (img, seg, label)

    else:
        yolo_weight = torch.load(weight_p)
        mask = yolo_weight["global_heatmap"].squeeze(0)
        yolo_patches = yolo_weight["patches"]
        patches = select_patches(yolo_patches)

        return (img, mask, patches, label)


def _read_image(path):
    try:
        return io.read_image(path).float() / 255.0
    except Exception:
        img = PIL.Image.open(path).convert("RGB")  # type: ignore
        img = TF.to_tensor(img)
        return img


def _erase(img, p=0.1):
    if torch.rand(1) < p:
        eraser = v2.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3))
        img = eraser(img)
    return img


def _erase_pair(img1, img2, p=0.1):
    if torch.rand(1) < p:
        i, j, h, w, v = v2.RandomErasing.get_params(
            img1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=None
        )
        img1 = TF.erase(img1, i, j, h, w, v)
        img2 = TF.erase(img2, i, j, h, w, v)
    return img1, img2

import numpy as np
from pathlib import Path
import os
from torch.utils.data import DataLoader

from pfe_crossvit_dual.training.dataset.dual_input_dataset import DualInputDataset
from pfe_crossvit_dual.training.utils.weight_functions import get_weight_function


def prepare_dataloaders(train_ds, val_ds, batch_size, num_workers):
    prefetch_factor = 2 if num_workers > 0 else None

    if num_workers != 0 and train_ds.precomputed:
        print("Dataset using cached data, num workers set to 0")
        num_workers = 0

    t_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=num_workers != 0,
        prefetch_factor=prefetch_factor,
    )
    v_ld = DataLoader(
        val_ds,
        batch_size=int(2 * batch_size),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers != 0,
        prefetch_factor=prefetch_factor,
    )
    return t_ld, v_ld


def get_data(
    data_dir,
    weight_function="linear",
    batch_size=16,
    num_workers=2,
    train_split=0.8,
    n_samples=None,
    shuffle=True,
    **dataset_kwargs,
):
    """Prépare les datasets et les loaders"""
    weight_fn = get_weight_function(weight_function)

    original_dir = Path(data_dir) / "original"
    img_paths = list(original_dir.rglob("*.jpg"))
    if shuffle:
        np.random.shuffle(img_paths)  # pyright: ignore[reportArgumentType]

    if n_samples is not None:
        img_paths = img_paths[:n_samples]

    train_max = int(train_split * len(img_paths))

    train_img_paths = img_paths[:train_max]
    val_img_paths = img_paths[train_max:]

    classes = sorted(os.listdir(original_dir))
    label_to_id = {l: i for i, l in enumerate(classes)}
    id_to_label = {i: l for i, l in enumerate(classes)}

    train_ds = DualInputDataset(
        label_to_id=label_to_id,
        img_paths=train_img_paths,
        is_train=True,
        weight_function=weight_fn,
        **dataset_kwargs,
    )
    val_ds = DualInputDataset(
        label_to_id=label_to_id,
        img_paths=val_img_paths,
        is_train=False,
        weight_function=weight_fn,
        **dataset_kwargs,
    )

    print(f"Train dataset length: {len(train_ds)}")
    print(f"Validation dataset length: {len(val_ds)}")

    return *prepare_dataloaders(train_ds, val_ds, batch_size, num_workers), id_to_label

from torch.utils.data import DataLoader, Subset
from pfe_crossvit_dual.training.dataset.dataset_vic import DualInputDataset
from pfe_crossvit_dual.training.utils.weight_functions import get_weight_function


def prepare_dataloaders(train_ds, val_ds, batch_size, num_workers):
    prefetch_factor = 2 if num_workers>0 else None
    t_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    v_ld = DataLoader(
        val_ds,
        batch_size=int(2 * batch_size),
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    return t_ld, v_ld


def get_data(
    data_dir,
    weight_function="linear",
    batch_size=16,
    num_workers=2,
    n_samples=None,
    **dataset_kwargs,
):
    """Prépare les datasets et les loaders"""
    weight_fn = get_weight_function(weight_function)

    train_ds = DualInputDataset(
        data_dir=data_dir,
        is_train=True,
        weight_function=weight_fn,
        **dataset_kwargs,
    )
    val_ds = DualInputDataset(
        data_dir=data_dir,
        is_train=False,
        weight_function=weight_fn,
        **dataset_kwargs,
    )

    if n_samples:
        train_ds = Subset(train_ds, range(min(n_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(n_samples, len(val_ds))))

    return prepare_dataloaders(train_ds, val_ds, batch_size, num_workers)

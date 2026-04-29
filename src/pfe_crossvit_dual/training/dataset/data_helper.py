from torch.utils.data import DataLoader, Subset
from pfe_crossvit_dual.training.dataset.dual_input_dataset import DualInputDataset
from pfe_crossvit_dual.training.utils.weight_functions import get_weight_function


def prepare_dataloaders(train_ds, val_ds, batch_size, num_workers):
    prefetch_factor = 2 if num_workers > 0 else None

    if num_workers !=0 and train_ds.precomputed:
        print("Dataset using cached data, num workers set to 0")
        num_workers = 0

    t_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers= num_workers != 0,
        prefetch_factor=prefetch_factor,
    )
    v_ld = DataLoader(
        val_ds,
        batch_size=int(2 * batch_size),
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers= num_workers != 0,
        prefetch_factor=prefetch_factor,
    )
    return t_ld, v_ld


def get_data(
    data_dir,
    weight_function="linear",
    batch_size=16,
    num_workers=2,
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

    return prepare_dataloaders(train_ds, val_ds, batch_size, num_workers)

import torch

from pfe_crossvit_dual.training.model.dual_crossvit import DualCrossVit
from pfe_crossvit_dual.training.model.model_loaders import (
    load_crossvit_pretrained_weights,
)


def load_training(model_fp: str, model, optimizer):
    print(f" > Resuming training of model at path: {model_fp}")

    checkpoint = torch.load(model_fp)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint.get("val_acc", 0.0)

    print(f" > Resuming at epoch {start_epoch + 1} with accuracy of {best_acc:.2f}%")


def instanciate_dualcrossvit(cross_vit, device, **model_kwargs):
    if "num_classes" not in model_kwargs:
        model_kwargs["num_classes"] = 2

    model = DualCrossVit(**model_kwargs)
    model = load_crossvit_pretrained_weights(model, cross_vit)
    model.to(device)

    return model

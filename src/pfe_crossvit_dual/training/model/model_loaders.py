import torch

from pfe_crossvit_dual.training.model.dual_crossvit import DualCrossVit
from pfe_crossvit_dual.training.model.dual_crossvit_yolo import DualCrossVitYolo
from pfe_crossvit_dual.training.model.crossvit_kwargs import CROSSVIT_KWARGS_MAP


DUALCROSSVIT_MAP = {
    "dual_crossvit": DualCrossVit,
    "dual_crossvit_yolo": DualCrossVitYolo,
}

# Clés KAN reconnues dans le config YAML — transmises au modèle si présentes
_KAN_KEYS = ("kan_mode", "kan_grid_size", "kan_bottleneck_dim", "kan_ffn_last_only")


def load_training(model_fp: str, model, lr):
    print(f" > Resuming training of model at path: {model_fp}")

    checkpoint = torch.load(model_fp)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint.get("val_acc", 0.0)

    print(f" > Resuming at epoch {start_epoch + 1} with accuracy of {best_acc:.2f}%")
    return model, optimizer


def instanciate_dualcrossvit(crossvit, model_name, device, **model_kwargs):
    crossvit_kwargs = CROSSVIT_KWARGS_MAP[crossvit]

    # Sépare les kwargs KAN du reste pour les logger proprement
    kan_kwargs = {k: model_kwargs.pop(k) for k in _KAN_KEYS if k in model_kwargs}

    if kan_kwargs:
        mode = kan_kwargs.get("kan_mode", "none")
        if mode != "none":
            print(f" > KAN mode : {mode!r}  (grid_size={kan_kwargs.get('kan_grid_size', 5)})")
        else:
            print(" > KAN mode : none (baseline)")

    model = DUALCROSSVIT_MAP[model_name](**model_kwargs, **crossvit_kwargs, **kan_kwargs)

    model = load_crossvit_pretrained_weights(model, crossvit)
    model.to(device)

    # Affiche un résumé KAN si le modèle expose kan_summary()
    if hasattr(model, "kan_summary"):
        print(model.kan_summary())

    return model


def load_crossvit_pretrained_weights(model, model_name="crossvit_15_224", dev=False):
    """
    Charge les poids pré-entraînés CrossViT.
    Ignore silencieusement les clés incompatibles (head, couches KAN sans équivalent pré-entraîné).
    """
    urls = {
        "crossvit_15_224": "https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_224.pth",
        "crossvit_18_224": "https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_224.pth",
        "crossvit_base_224": "https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_base_224.pth",
        "crossvit_small_224": "https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_small_224.pth",
        "crossvit_tiny_224": "https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_tiny_224.pth",
    }

    if model_name not in urls:
        raise ValueError(f"Unknonw model. Choices : {list(urls.keys())}")

    url = urls[model_name]
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    model_dict = model.state_dict()
    pretrained_dict = {}
    ignored_keys = []

    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
            else:
                ignored_keys.append(k)

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if dev:
        print("Weights loaded.")
        print(f"Missing keys : {ignored_keys}\n")

    return model
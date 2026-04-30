import torch

from pfe_crossvit_dual.training.model.dual_crossvit_ratio import DualCrossVitRatio
from pfe_crossvit_dual.training.model.dual_crossvit_yolo import DualCrossVitYolo
from pfe_crossvit_dual.training.model.crossvit_kwargs import CROSSVIT_KWARGS_MAP


DUALCROSSVIT_MAP = {
    "dual_crossvit_ratio": DualCrossVitRatio,
    "dual_crossvit_yolo": DualCrossVitYolo,
}


def load_training(model_fp: str, model, optimizer):
    print(f" > Resuming training of model at path: {model_fp}")

    checkpoint = torch.load(model_fp)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint.get("val_acc", 0.0)

    print(f" > Resuming at epoch {start_epoch + 1} with accuracy of {best_acc:.2f}%")


def instanciate_dualcrossvit(crossvit, model_name, device, **model_kwargs):
    crossvit_kwargs = CROSSVIT_KWARGS_MAP[crossvit]

    model = DUALCROSSVIT_MAP[model_name](**model_kwargs, **crossvit_kwargs)

    model = load_crossvit_pretrained_weights(model, crossvit)
    model.to(device)

    return model


def load_crossvit_pretrained_weights(model, model_name="crossvit_15_224", dev=False):
    """
    Load weights of pretrained model. Manages conflicts of dimensions (especially for the classification head where num_classes=2 instead of 1000)
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
    if dev:
        print(f"Loading weights from {model_name}.")

    checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    model_dict = model.state_dict()
    pretrained_dict = {}
    ignored_keys = []

    # filter keys by compatibility
    for k, v in state_dict.items():
        if k in model_dict:
            # Classification head won't be of same shape
            if v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
            else:
                ignored_keys.append(k)
        else:
            pass

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if dev:
        print("Weights loaded.")
        print(f"Missing keys : {ignored_keys}\n")

    return model

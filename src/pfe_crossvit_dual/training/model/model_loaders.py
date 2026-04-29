import torch
import torch.nn as nn
from PIL.Image import Image
import torchvision.transforms.functional as TF


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


def load_weights_for_symmetric_crossvit(
    symmetric_model: nn.Module, pretrained_model_name="crossvit_15_224"
):
    """
    This function loads the weights of a pretrained model of Crossvit into your symmetric model SMALL=LARGE
    so it can benefit from the pretraining while being a different architecture.

    Args :
        symmetric_model : should have a symmetric value for patch_size and branches_img_size
    """

    url = f"https://github.com/IBM/CrossViT/releases/download/weights-0.1/{pretrained_model_name}.pth"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    model_dict = symmetric_model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            new_state_dict[k] = v

        # We load the weights of the LARGE branch 'blocks.1' into the the SMALL Branch
        if "blocks.1" in k:
            k_sym = k.replace("blocks.1", "blocks.0")
            if k_sym in model_dict and v.shape == model_dict[k_sym].shape:
                new_state_dict[k_sym] = v

        elif "patch_embed.1" in k:
            k_sym = k.replace("patch_embed.1", "patch_embed.0")
            if k_sym in model_dict and v.shape == model_dict[k_sym].shape:
                new_state_dict[k_sym] = v

        elif "cls_token.1" in k:
            pass


def get_images_both_branches_as_tensors(
    img: Image, img_size_small: int, img_size_large: int, device, mean, std
):
    """
    returns both images as tensors ready to be used by the model's forward method.
    """
    img_tensor_small = TF.normalize(
        TF.to_tensor(
            TF.center_crop(TF.resize(img, int(img_size_small * 1.14)), img_size_small)  # type: ignore
        ),
        mean=mean,
        std=std,
    )
    img_tensor_small = img_tensor_small.unsqueeze(0).to(device)
    img_tensor_large = TF.normalize(
        TF.to_tensor(
            TF.center_crop(TF.resize(img, int(img_size_large * 1.14)), img_size_large)  # type: ignore
        ),
        mean=mean,
        std=std,
    )
    img_tensor_large = img_tensor_large.unsqueeze(0).to(device)

    return img_tensor_small, img_tensor_large

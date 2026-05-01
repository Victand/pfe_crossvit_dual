import torchvision.io as io
import torchvision.transforms.functional as F


def read_image(path):
    try:
        return io.read_image(str(path)).float() / 255.0
    except Exception:  # fall back for truncated images
        img = PIL.Image.open(str(path)).convert("RGB")  # type: ignore
        img = F.to_tensor(img)
        return img

import torchvision.io as io
import torchvision.transforms.functional as F
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(path):
    try:
        return io.read_image(str(path)).float() / 255.0
    except Exception:  # fall back for truncated images
        img = Image.open(str(path)).convert("RGB")
        img = F.to_tensor(img)
        return img

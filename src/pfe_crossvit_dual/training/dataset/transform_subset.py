from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import Image


class TransformSubset(Dataset):
    def __init__(self, subset, transforms) -> None:
        self.subset = subset
        self.

    def __getitem__(self, index):
        x_small, x_large, weight, label = self.subset[index]


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
        return x_small, x_large, weight, label
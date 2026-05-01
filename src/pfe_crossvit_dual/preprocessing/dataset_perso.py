import os
import glob
from pathlib import Path
import cv2
import numpy as np
from YOLOv7_ag.utils.datasets import letterbox

img_formats = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
    "webp",
    "mpo",
]  # acceptable image suffixes


class DatasetPerso:  # pour l'inférence
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # chemin absolu agnostique de l'OS
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))  # répertoire
        elif os.path.isfile(p):
            files = [p]  # fichiers
        else:
            raise Exception(f"ERROR: {p} does not exist")

        # Filtrer uniquement les images (en utilisant img_formats)
        images = [x for x in files if x.split(".")[-1].lower() in img_formats]

        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images  # Contient uniquement les chemins d'images
        self.nf = ni  # nombre de fichiers
        self.mode = "image"
        self.cap = None  # La variable 'cap' est inutile mais laissée si besoin.

        assert self.nf > 0, f"No images found in {p}. Supported formats are:\nimages: {img_formats}"

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        """

        :return:
        """
        if self.count == self.nf:
            raise StopIteration

        path = self.files[self.count]

        # Lecture de l'image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, "Image Not Found " + path
        # print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize (l'appel à la fonction 'letterbox' est conservé)
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR vers RGB, vers 3xHxW
        img = np.ascontiguousarray(img)

        # Signature simplifiée (on retire self.cap)
        # Chemin absolu de l'image
        return path, img, img0

    # new_video est supprimée

    def __len__(self):
        return self.nf  # nombre de fichiers


from pathlib import Path
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnTF

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from scripts.training import plot_weight_example
from scripts.utils.weight_functions import linear_

ALL_TRANSFORMS = ["random_crop","hflip","color_jitter","random_erase"]
DATA_PATH_STR = "data\created_data"

class DualInputDataset(Dataset):
    """
    Custom Dataset to manage a dual input data of non segmented and segmented herb image.

    Transformation for data augmentation is managed internally because of the need for synchornized transforms between original and segmented herb images.

    List of all transforms : {all_transforms}
    Use method set_active_transforms() to choose which combinaison you prefer.
    """
    def __init__(self, data_dir: str = DATA_PATH_STR, is_train: bool = True, image_size=(240, 240), patch_size=(16, 16),
                 classes=("class1", "class2"), paths=("original", "segmented"), check_data=False, pounderation=False,
                 weight_function=linear_, use_yolo_weights=False):  # <-- NOUVEAU: paramètre use_yolo_weights
        """
        ... (tes commentaires inchangés) ...
        """
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.image_size = image_size
        self.patch_size = patch_size
        self.classes = classes
        self.paths = paths
        self.pounderation = pounderation
        self.weight_function = weight_function
        self.numbranches = 2

        # NOUVEAU : On enregistre le paramètre
        self.use_yolo_weights = use_yolo_weights

        # Standard ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # List of transformations
        self.all_transforms = ALL_TRANSFORMS
        self.active_transforms = {transform: True for transform in self.all_transforms}

        # [(path_original, path_detoure, label_int), ...]
        self.samples = []
        self.classes_count = {self.classes[0]: 0, self.classes[1]: 0}
        self.load_samples()

        if check_data:
            samples_to_show = find_two_samples(self.samples)
            plot_samples(samples_to_show, classes=self.classes, paths=self.paths,
                         suptitle=f'A sample image for each class --- data_path : {self.data_dir}\nConfiguration : {self.paths}')


    def load_samples(self):
        """
        Scan folders to load the samples in the dataset.
        Awaited folder structure :
            data/
            ├── original/              <-- Branch 1
            │   ├── class1/       
            │   │   ├── plante_A.jpg
            │   │   └── plante_B.jpg
            │   └── class2/       
            │       ├── plante_C.jpg
            │       └── plante_D.jpg
            │
            └── segmented/             <-- Branch 2 
                ├── class1/       
                │   ├── plante_A.jpg   
                │   └── plante_B.jpg
                └── class2/
                    ├── plante_C.jpg
                    └── plante_D.jpg
        """
        phase = "train" if self.is_train else "val"

        for label_int, _class in enumerate(self.classes):
            original_path, segmented_path = self.data_dir / phase / self.paths[0] / _class, self.data_dir / phase / self.paths[1] / _class

            if not (original_path.exists() and segmented_path.exists()):
                raise FileNotFoundError(f"{original_path} or {segmented_path} folder does not exist.")

            for original_img_p in original_path.iterdir():
                image = original_img_p.name

                segmented_img_p = segmented_path / image
                if not segmented_img_p.exists():
                    print(f"Non detoured image : {image} does not match with any detoured image.")
                    continue

                if self.use_yolo_weights:
                    weight_p = original_img_p.parent / f"{original_img_p.stem}_weights.pt"
                    # Si le fichier .pt n'existe pas, on ignore cette image !
                    if not weight_p.exists():
                        continue
                
                self.samples.append((original_img_p,segmented_img_p,label_int))
                self.classes_count[self.classes[label_int]] += 1
        
        # Shuffle the samples for better representation of classes
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)



    def __getitem__(self, idx):
        original_image_p, segmented_image_p, label_int = self.samples[idx]
        original_image = Image.open(original_image_p).convert('RGB')
        segmented_image = Image.open(segmented_image_p).convert('RGB')

        original_image, segmented_image = self.synchronized_transform(original_image, segmented_image)

        # --- NOUVEAU : GESTION DES POIDS YOLO ---
        if self.use_yolo_weights:
            # On cherche le fichier de poids avec le même nom, mais finissant par _weights.pt
            # Exemple : si l'image est "plante_A.jpg", on cherche "plante_A_weights.pt"
            # On suppose que ce fichier est dans le même dossier que l'image originale
            weight_p = original_image_p.parent / f"{original_image_p.stem}_weights.pt"

            if weight_p.exists():
                weights = torch.load(weight_p)
            else:
                # Sécurité au cas où tu as oublié de générer les poids pour une image
                print(f"⚠️ Poids YOLO introuvables pour {original_image_p.name}, utilisation de poids neutres.")
                num_patches = (self.image_size[0] // self.patch_size[0]) ** 2
                weights = torch.ones((num_patches, 1))

        elif self.pounderation:
            # Ton ancienne méthode avec la segmentation
            weights = self.patches_weights(segmented_image, self.patch_size[0], self.weight_function)

        else:
            # If no pounderation we will detect an empty tensor
            weights = torch.empty(0)
        # ----------------------------------------

        original_image, segmented_image = normalize_and_erase(self, original_image, segmented_image)

        # On retourne les poids (YOLO, Segmentation, ou vide) !
        return original_image, segmented_image, label_int, weights


    def patches_weights(self, segmented_tensor, patch_size:int, f=lambda x: x + 1e-7):
        """
        Returns the weight vector of all patches weights in the shape (num_patches,1),
        so the vector is (batch_size, num_patches, 1) in output of the dataloader.
        
        """
        # Creates a binary mask from the segmented image
        # if background : Channels = (0,0,0)
        # unfold needs a 4D tensor so we artifially add a Batch dimension with unsqueeze
        mask = (torch.sum(segmented_tensor,dim=0)>0).float().unsqueeze(dim=0).unsqueeze(0)
        # Extracts patches
        # (1,1,H,W) -> (num_pixels, num_patches)
        patches = nnTF.unfold(mask, kernel_size=patch_size,stride=patch_size).squeeze(0)
        # Measures ratios for the image
        ratios = torch.mean(patches, dim=0)

        num_patches = ratios.numel()

        # We consider the case where there is only background
        if not ratios.sum() > 0:
            w_norm = torch.ones(num_patches)
        else:
            # Computes weights
            w_p = f(ratios)

            # Normalization : unitary mean
            epsilon = 1e-8
            sum_w = w_p.sum()
            w_norm = w_p*num_patches/(sum_w + epsilon) 

        # We return a (num_patches,1) shaped tensor
        w_norm = torch.unsqueeze(w_norm, dim=1)
        return w_norm
    

    def set_active_transforms(self, active_transforms_list):
        """
        Choose which transforms shoold be applied within this list : {all_transforms}
        By default every transforms are applied.
        """
        self.active_transforms = {transform:False for transform in self.all_transforms}
        for tf in active_transforms_list:
            if not (tf in self.all_transforms):
                print(f"{tf} transform name does not exist.")
                self.active_transforms = {transform:True for transform in self.all_transforms}
                break
            self.active_transforms[tf] = True


    def synchronized_transform(self, img1, img2):
        """
        Applies same transformations on both images.
        Without normalization and random erase.
        """
        a_transforms = self.active_transforms

        # Upsizing to respect aspect ratio when cropping
        new_size = (int(self.image_size[0] * 1.14),int(self.image_size[1] * 1.14))
        img1 = TF.resize(img1, new_size[0])
        img2 = TF.resize(img2, new_size[1])

        if self.is_train:

            # Random crop
            if a_transforms["random_crop"]:
                i, j, h, w = transforms.RandomCrop.get_params(
                    img1, output_size=(self.image_size[0], self.image_size[0]))

                scale_factor = self.image_size[1] / self.image_size[0]

                img1 = TF.crop(img1, i, j, h, w)

                i2 = int(i * scale_factor)
                j2 = int(j * scale_factor)
                h2 = int(h * scale_factor)
                w2 = int(w * scale_factor)

                img2 = TF.crop(img2, i2, j2, h2, w2)

            # Random horizontal flip
            if a_transforms["hflip"]:
                if random.random() > 0.5:
                    img1 = TF.hflip(img1)
                    img2 = TF.hflip(img2)

            # Color jitter 
            if a_transforms["color_jitter"]:
                jitter_params = transforms.ColorJitter.get_params(
                    [0.6, 1.4],
                    [0.6, 1.4],
                    [0.6, 1.4],
                    [-0.1,0.1]
                    )
                fn_idx, brightness, contrast, saturation, hue = jitter_params

                for fn_id in fn_idx:
                    if fn_id == 0 and brightness is not None:
                        img1 = TF.adjust_brightness(img1, brightness)
                        img2 = TF.adjust_brightness(img2, brightness)
                    elif fn_id == 1 and contrast is not None:
                        img1 = TF.adjust_contrast(img1, contrast)
                        img2 = TF.adjust_contrast(img2, contrast)
                    elif fn_id == 2 and saturation is not None:
                        img1 = TF.adjust_saturation(img1, saturation)
                        img2 = TF.adjust_saturation(img2, saturation)
                    elif fn_id == 3 and hue is not None:
                        img1 = TF.adjust_hue(img1, hue)
                        img2 = TF.adjust_hue(img2, hue)
            
        if (not a_transforms["random_crop"]) or (not self.is_train):
            img1 = TF.center_crop(img1, self.image_size[0])
            img2 = TF.center_crop(img2, self.image_size[1])

        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        return img1, img2
    
    
    
# Change docstring
DualInputDataset.__doc__ = DualInputDataset.__doc__.format(all_transforms=ALL_TRANSFORMS)
DualInputDataset.set_active_transforms.__doc__ = DualInputDataset.set_active_transforms.__doc__.format(all_transforms=ALL_TRANSFORMS)

#####################################################################################

def normalize_and_erase(dualdataset:DualInputDataset, img1:torch.Tensor, img2:torch.Tensor):
    """
    Applies normalization and random erase if this transformation is accepted byt the instance dataset.
    """
    img1 = TF.normalize(img1, dualdataset.mean, dualdataset.std)
    img2 = TF.normalize(img2, dualdataset.mean, dualdataset.std)

    # Random erasing
    if dualdataset.is_train and dualdataset.active_transforms["random_erase"]:
        p_erase = 0.1 
        
        if random.random() < p_erase:
            i, j, h, w, v = transforms.RandomErasing.get_params(
                img1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=None
            )
            
            ratio_h = dualdataset.image_size[1] / dualdataset.image_size[0]
            ratio_w = dualdataset.image_size[1] / dualdataset.image_size[0]
            
            img1 = TF.erase(img1, i, j, h, w, v, inplace=False)
            
            i2, h2 = int(i * ratio_h), int(h * ratio_h)
            j2, w2 = int(j * ratio_w), int(w * ratio_w)
            v2 = TF.resize(v, [h2, w2], antialias=True)
            
            img2 = TF.erase(img2, i2, j2, h2, w2, v2, inplace=False)

    return img1, img2

def prepare_dataloaders(train_dataset,val_dataset,batch_size,num_workers):
    """
    Instanciates train and val dataloaders.
    """
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(2*batch_size),
        num_workers=num_workers,
        shuffle=False
    )

    return data_loader_train,data_loader_val

#####################################################################################

def plot_samples(samples, classes, paths, suptitle=''):
    """
    Plot the samples with original image, segmented image and its class.
    samples List(Tuple(str,str,int)) : [(path_original, path_detoure, label_int), ...]
    """
    n = len(samples)

    fig, axes = plt.subplots(n,2)

    for i,sample in enumerate(samples):
            for j, path in enumerate(sample[:-1]):
                img = Image.open(path).transpose(Image.ROTATE_90)

                # Resize
                img.thumbnail((1000,1000))

                axes[i,j].imshow(img)
                if j==0:
                    axes[i,j].set(title=f"Picture : {path.stem}\nLabel = {sample[2]} / Class = {classes[sample[2]]}")
                    axes[i,j].set(ylabel=f"Branch 1 : {paths[0]}")
                else:
                    axes[i,j].set(ylabel=f"Branch 2 : {paths[1]}")

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_samples_with_weights(samples, classes, paths, alpha_config, suptitle=''):
    """
    Affiche l'image originale, l'image segmentée et la heatmap YOLO
    superposée à l'image originale pour vérifier l'alignement.
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))

    if n == 1: axes = np.expand_dims(axes, axis=0)

    for i, sample in enumerate(samples):
        orig_p, seg_p, label_int = sample

        # 1. Image Originale
        img_orig = Image.open(orig_p).transpose(Image.ROTATE_90)
        axes[i, 0].imshow(img_orig)
        axes[i, 0].set_title(f"Class: {classes[label_int]}")
        axes[i, 0].set_ylabel(f"Sample {i + 1}")

        # 2. Image Segmentée
        img_seg = Image.open(seg_p).transpose(Image.ROTATE_90)
        axes[i, 1].imshow(img_seg)
        axes[i, 1].set_title("Segmented")

        # 3. Superposition (Overlay) Image + Heatmap
        weight_p = orig_p.parent / f"{orig_p.stem}_weights.pt"
        if weight_p.exists():
            # Affichage de l'image de fond
            axes[i, 2].imshow(img_orig)

            # Chargement et mixage des poids
            w_multi = torch.load(weight_p)
            w_mixed = (w_multi * alpha_config.view(1, -1, 1, 1)).sum(dim=1).squeeze()

            # On redimensionne la heatmap (14x14) à la taille de l'image pour le plot
            # 'extent' permet de caler la heatmap exactement sur les bords de l'image
            im = axes[i, 2].imshow(w_mixed.numpy(),
                                   cmap='hot',
                                   alpha=0.5,  # 50% de transparence pour voir l'image derrière
                                   extent=[0, img_orig.size[0], img_orig.size[1], 0])

            plt.colorbar(im, ax=axes[i, 2])
            axes[i, 2].set_title("YOLO Overlay (Transparency)")
        else:
            axes[i, 2].text(0.5, 0.5, "No .pt found", ha='center')

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def find_two_samples(samples):
    """
    Find one random sample for each label/class.
    Args:
        samples : [(original_img_p,segmented_img_p,label_int),...]
    """
    s1 = np.random.randint(0,len(samples))
    sample1=samples[s1]
    label1 = sample1[2]
    s2 = np.random.randint(0,len(samples))
    sample2=samples[s2]
    while sample2[2]==label1:
        s2 = np.random.randint(0,len(samples))
        sample2=samples[s2]
    return [sample1, sample2]
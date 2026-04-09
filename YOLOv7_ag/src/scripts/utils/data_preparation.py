import shutil
import time
from pandas import DataFrame
from pathlib import Path
import random

from converters import numbersToDf

### TEMPORARY TEST ###
labels_df = numbersToDf().drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)
print(f"Aperçu Dataframe = \n{labels_df.head(n=8)}")
DATA_PATH_STR = 'data/created_data'
######################

def dirStructAsTorchvisionLayout(labels:DataFrame, data_root_path:Path, original_images_path:Path, segmented_images_path:Path, train_split:float=1.0, classes=("class1","class2")):
    """
    Create a directory structure as expected by torchvision.datasets for a binary classification where detouring is used : \\

    data/
    ├── train/               
    │   ├── original/           <- original branch
    │   │   ├── class1/
    │   │   └── class2/
    │   └── segmented/          <- segmented branch
    │       ├── class1/
    │       └── class2/
    │
    └── val/                  
        ├── original/
        │   ├── class1/
        │   └── class2/
        └── segmented/
            ├── class1/
            └── class2/

    For this, class directories will be created and images will be stored following the label given.

    Labels should be such that value is 0.0 if class1 and 1.0 if class2.

    Args:
        labels (pandas.DataFrame) : stored labels.
        data_root_path (pathlib.Path) : where you want to create you directory structure.
        original_images_path (pathlib.Path) : path to the original images, jpg images
        segmented_images_path (pathlib.Path) : path to the detoured images, jpg images
        train_split (float) : the expected split between train data and validation data. ]0,1[
        classes ((str,str)) : the two classe name.
    """

    ## Creation of the arborescence with empty folders

    print("\nCreating folders...")

    modes = ["train","val"]
    branches = ["original","segmented"]

    for mode_name in modes:
        p = data_root_path / mode_name
        for branch_path in branches:
            for class_ in classes:
                class_path = p / branch_path / class_
                try:
                    class_path.mkdir(parents=True)
                except FileExistsError as e:
                    print("Folder already created.")
                except FileNotFoundError as e:
                    print(e)


    ## Finding labeled data

    labeled_images_dict = {}

    # Execution time
    start_time = time.perf_counter()
        
    # Tracking unlabelled data
    no_labels_count = 0
    ## Storing images based on class
    df = labels.copy()
    col1, col2 = df.columns
    df[col1] = df[col1].astype(str).str.strip()

    # Dict of the label dataframe
    code_epines_dict = dict(zip(df[col1],df[col2]))
    

    all_images = list(original_images_path.glob("*.jpg"))

    print(f"Found {len(all_images)} images in source folder.")

    class1_count = 0
    for original_img_path in all_images:
        # image name without suffix .jpg
        code = original_img_path.stem.strip()
        
        try:
            # if this image is labeled i.e. appears in dataframe
            col2_value = code_epines_dict[code]
            is_class1 = True if 0.0==float(col2_value) else False
            labeled_images_dict[code] = {
                "is_class1": is_class1,
                "original_img_path": original_img_path
                }
            if is_class1:
                class1_count += 1
        except KeyError:
            no_labels_count+=1

    ## Copying labeled data into the new folders structure
    
    print("Copying labeled data...")


    n_labeled_data = len(labeled_images_dict)
    n_train = int(train_split*n_labeled_data)
    n_val = n_labeled_data - n_train

    # We mesure the ratio between the two classes
    # We need this ratio to be respected in both train and val sets.
    class1_ratio = class1_count/n_labeled_data

    class1_train_max = int(class1_ratio*n_train)+1
    class2_train_max = n_train - class1_train_max

    train_max = (class1_train_max,class2_train_max)
    train_count = [0,0]
    val_count = [0,0]

    keys = list(labeled_images_dict.keys())
    for code in keys:
        is_class1 = labeled_images_dict[code]['is_class1']
        
        original_img_path = labeled_images_dict[code]['original_img_path']
        segmented_img_path = segmented_images_path / f"{code}.jpg"

        # Security before copying data
        if not segmented_img_path.exists():
            print(f"Following image does not have segmented correspondant : {original_img_path.name}")
            continue
        
        class_num = 0 if is_class1 else 1

        current_class = classes[class_num]

        # Filling the train folder while not full and while class ratio is respected
        if sum(train_count) < n_train and train_count[class_num] < train_max[class_num]:
            current_mode = modes[0]
            train_count[class_num] += 1
        # Else filling val folder
        else:
            current_mode = modes[1]
            val_count[class_num] += 1
            
        original_dest_path = data_root_path / f"{current_mode}/{branches[0]}/{current_class}"
        segmented_dest_path = data_root_path / f"{current_mode}/{branches[1]}/{current_class}"

        shutil.copy(original_img_path, original_dest_path)
        shutil.copy(segmented_img_path, segmented_dest_path)


    end_time = time.perf_counter()

    print(f"\n{n_labeled_data} elements copied to {data_root_path}. Execution time = {end_time-start_time:.2f}s\nNumber of images without correspondant label = {no_labels_count}")
    print("-"*20)
    print(f"Train split : {train_split*100}%    No epines/epines ratio : {int(class1_ratio*100)}/{100-int(class1_ratio*100)}")
    print(f"train epines : {train_count[1]}, train sans epines : {train_count[0]}")
    print(f"val epines : {val_count[1]}, val sans epines : {val_count[0]}")
    print(f"Total = train: {n_train}    val: {n_val}")


### DEMO ########
original_imgs_path = Path("data/full_data/mission_herbonaute_2000")
segmented_imgs_path= Path("data/full_data/mission_herbonaute_2000_seg_black")

dirStructAsTorchvisionLayout(labels_df, data_root_path=Path(DATA_PATH_STR), original_images_path=original_imgs_path, segmented_images_path=segmented_imgs_path, train_split=0.8, classes=("no_epines","epines"))
#################


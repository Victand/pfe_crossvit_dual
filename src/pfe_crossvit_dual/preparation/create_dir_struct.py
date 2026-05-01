import shutil
import time
from pandas import DataFrame
from pathlib import Path
from tqdm import tqdm
import re
import os


def create_genera_dir_struct(
    output_dir: Path,
    original_dir: Path,
    segmented_dir: Path,
):
    """Same as create_thorns_dir_struct but for genera dataset, with genera classes"""

    branches = ["original", "segmented"]

    # get classes
    # get all image ids (eg "castanea146")
    img_ids = [fn.split(".")[0] for fn in os.listdir(original_dir) if fn.endswith(".jpg")]
    # strip number from image ids
    labels = [re.sub(r"\d+", "", img_id) for img_id in img_ids]
    # get unique classes
    classes = list(set(labels))

    # create directories
    for branch_path in branches:
        for class_ in classes:
            class_path = output_dir / branch_path / class_
            try:
                class_path.mkdir(parents=True)
            except FileExistsError:
                print("Folder already created.")
            except FileNotFoundError as e:
                print(e)

    # move files
    for img_id, label in zip(tqdm(img_ids, desc="copying files"), labels):
        original_img_fp = original_dir / f"{img_id}.jpg"
        segmented_img_fp = segmented_dir / f"{img_id}.jpg"

        original_dest_dir = output_dir / "original" / label
        segmented_dest_dir = output_dir / "segmented" / label

        shutil.copy(original_img_fp, original_dest_dir)
        shutil.copy(segmented_img_fp, segmented_dest_dir)

    print(f"Copied {len(img_ids * 2)} images")


def create_thorns_dir_struct(
    labels: DataFrame,
    data_root_path: Path,
    original_images_path: Path,
    segmented_images_path: Path,
):
    """
    Create a directory structure as expected by torchvision.datasets for a binary classification where detouring is used : \\

    data/           
    ├── original/           <- original branch
    │   ├── class1/
    │   └── class2/
    └── segmented/          <- segmented branch
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

    classes = ["no_thorns", "thorns"]
    branches = ["original", "segmented"]

    for branch_path in branches:
        for class_ in classes:
            class_path = data_root_path / branch_path / class_
            try:
                class_path.mkdir(parents=True)
            except FileExistsError:
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
    code_epines_dict = dict(zip(df[col1], df[col2]))

    all_images = list(original_images_path.glob("*.jpg"))

    print(f"Found {len(all_images)} images in source folder.")

    class1_count = 0
    for original_img_path in all_images:
        # image name without suffix .jpg
        code = original_img_path.stem.strip()

        try:
            # if this image is labeled i.e. appears in dataframe
            col2_value = code_epines_dict[code]
            is_class1 = True if 0.0 == float(col2_value) else False
            labeled_images_dict[code] = {
                "is_class1": is_class1,
                "original_img_path": original_img_path,
            }
            if is_class1:
                class1_count += 1
        except KeyError:
            no_labels_count += 1

    ## Copying labeled data into the new folders structure

    print("Copying labeled data...")

    n_labeled_data = len(labeled_images_dict)

    # We mesure the ratio between the two classes
    # We need this ratio to be respected in both train and val sets.
    class1_ratio = class1_count / n_labeled_data

    keys = list(labeled_images_dict.keys())

    for code in keys:
        is_class1 = labeled_images_dict[code]["is_class1"]

        original_img_path = labeled_images_dict[code]["original_img_path"]
        segmented_img_path = segmented_images_path / f"{code}.jpg"

        # Security before copying data
        if not segmented_img_path.exists():
            print(
                f"Following image does not have segmented correspondant : {original_img_path.name}"
            )
            continue

        class_num = 0 if is_class1 else 1
        current_class = classes[class_num]

        original_dest_path = data_root_path / branches[0] / current_class
        segmented_dest_path = data_root_path / branches[1] / current_class

        shutil.copy(original_img_path, original_dest_path)
        shutil.copy(segmented_img_path, segmented_dest_path)

    end_time = time.perf_counter()

    print(
        f"\n{n_labeled_data} elements copied to {data_root_path}. Execution time = {end_time - start_time:.2f}s\nNumber of images without correspondant label = {no_labels_count}"
    )
    print("-" * 20)
    print(f"no_thorns/thorns ratio : {int(class1_ratio * 100)}/{100 - int(class1_ratio * 100)}")
    print(f"No Thorns count: {class1_count}    Thorns count: {n_labeled_data - class1_count}")
    print(f"No labels count: {no_labels_count}")

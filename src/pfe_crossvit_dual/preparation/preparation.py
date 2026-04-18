from pathlib import Path
import os
import argparse

from pfe_crossvit_dual.preparation.converters import numbersToDf
from pfe_crossvit_dual.preparation.create_dir_struct import dirStructAsTorchvisionLayout
from pfe_crossvit_dual.constants.paths import DATA_DIR


def data_preparation_pipeline(input_dir):
    """Transforms the thorns dataset into a classic torch directory layout"""
    output_dir = DATA_DIR

    original_imgs_path = Path(os.path.join(input_dir, "mission_herbonaute_2000"))
    segmented_imgs_path = Path(
        os.path.join(input_dir, "mission_herbonaute_2000_seg_black")
    )
    labels_path = Path(os.path.join(input_dir, "Data_v2.numbers"))

    labels_df = (
        numbersToDf(labels_path)
        .drop_duplicates(subset=["code"], keep="first")
        .reset_index(drop=True)
    )
    print(f"Aperçu Dataframe = \n{labels_df.head(n=8)}")

    dirStructAsTorchvisionLayout(
        labels_df,
        data_root_path=Path(output_dir),
        original_images_path=original_imgs_path,
        segmented_images_path=segmented_imgs_path,
        train_split=0.8,
        classes=("no_epines", "epines"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path to thorns dataset"
    )
    args = parser.parse_args()

    data_preparation_pipeline(args.data_dir)

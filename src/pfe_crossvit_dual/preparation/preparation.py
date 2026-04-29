from pathlib import Path
import argparse

from pfe_crossvit_dual.preparation.converters import numbersToDf
from pfe_crossvit_dual.preparation.create_dir_struct import (
    create_thorns_dir_struct,
    create_genera_dir_struct,
)


def data_preparation_pipeline(dataset, input_dir, output_dir):
    """Transforms the dataset into a classic torch directory layout"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if dataset == "genera":
        original_dir = input_dir / "images"
        segmented_dir = input_dir / "masks"

        create_genera_dir_struct(output_dir, original_dir, segmented_dir)

    elif dataset == "thorns":
        original_dir = input_dir / "mission_herbonaute_2000"
        segmented_dir = input_dir / "mission_herbonaute_2000_seg_black"
        labels_path = input_dir / "Data_v2.numbers"

        labels_df = (
            numbersToDf(labels_path)
            .drop_duplicates(subset=["code"], keep="first")
            .reset_index(drop=True)
        )
        print(f"Aperçu Dataframe = \n{labels_df.head(n=8)}")

        create_thorns_dir_struct(
            labels_df,
            data_root_path=output_dir,
            original_images_path=original_dir,
            segmented_images_path=segmented_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, help="Dataset type, either 'thorns' or 'genera'."
    )
    parser.add_argument("--input_dir", type=str, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()

    data_preparation_pipeline(args.dataset, args.input_dir, args.output_dir)

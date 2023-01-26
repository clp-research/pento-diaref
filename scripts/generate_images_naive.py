import argparse
import os
from typing import Tuple

from pentodiaref.data.generation.visuals import collect_and_store_images


def cli(split_name: str, data_dir: str, image_size: Tuple[int, int], force_reindex: bool = False,
        dry_run: bool = False):
    """
        Generate the images for the 'random' splits into h5py files.
    """
    if dry_run:
        print("Dry run detected")

    if force_reindex:
        print("force_reindex detected")

    # There are no categories for random sampling dataset
    category_name = None

    if split_name != "all":
        print(f"Split name '{split_name}' detected")
        collect_and_store_images(image_size, data_dir, split_name, category_name, force_reindex, dry_run)
        return

    print("Split name 'all' detected, will generate images for all splits")
    split_names = ["train", "val", "test"]
    for split_name in split_names:
        collect_and_store_images(image_size, data_dir, "data_" + split_name, category_name, force_reindex, dry_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--data_dir', type=str,
                        default="/data/pento_diaref/didact", help="Default: /data/pento_diaref/didact")
    parser.add_argument("-i", '--image_size', nargs=2, default=[224, 224], type=int,
                        help="Default: 224,224")
    parser.add_argument("-s", '--split_name', required=True, type=str,
                        help="['all', 'data', 'val', 'test']")
    parser.add_argument("-r", '--dry_run', action="store_true")
    parser.add_argument("-f", '--force_reindex', action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.data_dir):
        raise Exception("data_dir does not exist. Please create it and try again.")
    image_size = int(args.image_size[0]), int(args.image_size[1])
    cli(args.split_name,
        args.data_dir,
        tuple(args.image_size),
        args.force_reindex,
        args.dry_run)

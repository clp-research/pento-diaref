import itertools
import os
import argparse
from typing import Tuple

from pentodiaref.data.generation.visuals import collect_and_store_images


def cli(split_name: str, data_dir: str, category_name: str, image_size: Tuple[int, int],
        force_reindex: bool = False, dry_run: bool = False):
    """
        Generate the images for the 'didact' splits into h5py files.

        split_name and category_name will be concatenated e.g. to ho-color_val

        For 'data' there is no category.
    """
    if dry_run:
        print("Dry run detected")

    if force_reindex:
        print("force_reindex detected")

    if split_name.startswith("data"):
        category_name = None
        print(f"Split name '{split_name}' detected (no category)")
        collect_and_store_images(image_size, data_dir, split_name, category_name, force_reindex, dry_run)
        return

    if split_name != "all":
        print(f"Category '{category_name}' and split name '{split_name}' detected")
        collect_and_store_images(image_size, data_dir, split_name, category_name, force_reindex, dry_run)
        return

    print("Split name 'all' detected, will generate images for all splits")
    split_combs = list(itertools.product(['ho-color', 'ho-pos', "ho-uts"], ["val", "test"]))
    split_combs.append((None, "data_train"))
    split_combs.append((None, "data_val"))
    split_combs.append((None, "data_test"))
    for (category_name, split_name) in split_combs:
        collect_and_store_images(image_size, data_dir, split_name, category_name, force_reindex, dry_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--data_dir', type=str,
                        default="/data/pento_diaref/didact", help="Default: /data/pento_diaref/didact")
    parser.add_argument("-i", '--image_size', nargs=2, default=[224, 224], type=int,
                        help="Default: 224,224")
    parser.add_argument("-s", '--split_name', required=True,
                        type=str, help="['all', 'data', 'val', 'test']")
    parser.add_argument("-c", '--category_name', type=str,
                        help="If to produce only for a specific category ['ho-color', 'ho-pos', 'ho-uts']")
    parser.add_argument("-r", '--dry_run', action="store_true")
    parser.add_argument("-f", '--force_reindex', action="store_true",
                        help="Align the annotation id with the position in the h5py file. "
                             "For example, the validation samples might have arbitrary ids, but we create an h5py file"
                             "where the images of the samples are indexed by their position.")
    args = parser.parse_args()
    if not os.path.exists(args.data_dir):
        raise Exception("data_dir does not exist. Please create it and try again.")
    image_size = int(args.image_size[0]), int(args.image_size[1])
    cli(args.split_name,
        args.data_dir,
        args.category_name,
        image_size,
        args.force_reindex,
        args.dry_run)

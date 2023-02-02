import argparse
import os
from typing import Tuple
import random
import numpy as np

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

    if split_name is None:
        collect_and_store_images(image_size, data_dir, "data", split_name, force_reindex, dry_run)
    else:
        collect_and_store_images(image_size, data_dir, "data", split_name, force_reindex, dry_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--data_dir', type=str,
                        default="/data/pento_diaref/naive", help="Default: /data/pento_diaref/naive")
    parser.add_argument("-i", '--image_size', nargs=2, default=[224, 224], type=int,
                        help="Default: 224,224")
    parser.add_argument("-s", '--split_name', type=str,
                        help="Optionally a specific split: ['train', 'val', 'test']. Default: 'all'")
    parser.add_argument("-S", '--seed', type=int, default=None)
    parser.add_argument("-r", '--dry_run', action="store_true")
    parser.add_argument("-f", '--force_reindex', action="store_true",
                        help="Align the annotation id with the position in the h5py file. "
                             "For example, the validation samples might have arbitrary ids, but we might want to "
                             "create an h5py file where the images of the samples are indexed by their position.")
    args = parser.parse_args()
    if not os.path.exists(args.data_dir):
        raise Exception("data_dir does not exist. Please create it and try again.")
    image_size = int(args.image_size[0]), int(args.image_size[1])
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    cli(args.split_name,
        args.data_dir,
        tuple(args.image_size),
        args.force_reindex,
        args.dry_run)

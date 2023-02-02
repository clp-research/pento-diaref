import json
import os
from collections import defaultdict
from typing import List, Tuple

import h5py
from golmi.contrib.pentomino.objects import BoardPlotContext, Board
from golmi.server.grid import GridConfig
from tqdm import tqdm

from pentodiaref.data.generation.types import Annotation


def collect_and_store_images(target_size: Tuple[int, int], data_dir: str,
                             category_name: str, split_name: str = None,
                             force_reindex=False, dry_run=False):
    assert category_name is not None, "category_name must be given"
    if category_name == "data":  # we might want to generate the images for all samples altogether
        print("Detected 'data' category")
        annotations = []
        if split_name is None:
            print("Split name 'all' detected, will generate images for all splits together")
            split_names = ["train", "val", "test"]
        else:
            print(f"Split name '{split_name}' detected, will generate images only for this one")
            split_names = [split_name]
        for suffix in split_names:
            split_annotations = Annotation.load(data_dir, "data_" + suffix, resolve=True)
            annotations.extend(split_annotations)
        fn = "data"
        print(f"Loaded {len(annotations)} annotations in total")
    else:  # holdout categories
        assert split_name is not None, "split_name must be given"
        fn = f"{category_name}_{split_name}"
        annotations = Annotation.load(data_dir, fn, resolve=True)
    if force_reindex:  # this might be necessary, when we split a larger set of samples
        print("Force re-index of annotations to be aligned with h5py index")
        for idx, a in enumerate(annotations):
            a.anno_id = idx
    annos_with_bboxes = collect_images_and_bboxes(annotations, data_dir, fn, target_size, dry_run=dry_run)
    if category_name == "data":  # store in separate files again
        by_split_name = defaultdict(list)
        for a in annos_with_bboxes:
            by_split_name[a.split_name].append(a)
        if split_name is None:
            split_names = ["train", "val", "test"]
        else:
            split_names = [split_name]
        for suffix in split_names:
            if dry_run:
                print(f"Dry-run: Store annotations to data_" + suffix)
            else:
                fn = "data_" + suffix
                Annotation.store(by_split_name[fn], fn, data_dir)
    else:
        if dry_run:
            print(f"Dry-run: Store annotations to " + fn)
        else:
            Annotation.store(annos_with_bboxes, fn, data_dir)


def collect_images_and_bboxes(annotations: List[Annotation], data_dir: str, images_filename: str, image_size,
                              verbose=False, dry_run=False):
    # We directly store into the state file line by line to save memory and compute
    state_file_path = os.path.join(data_dir, images_filename + ".states")
    if os.path.exists(state_file_path):
        os.remove(state_file_path)
    state_file = open(state_file_path, "a")

    piece_groups_by_id = dict([(a.group_id, a.group) for a in annotations])
    print(f"Detected {len(piece_groups_by_id)} piece groups (to create images for)")

    if dry_run:  # we cannot store the images because the selection might not contain an ordered range (0..N)
        return annotations

    bounding_boxes_by_group_id = dict()
    file_path = os.path.join(data_dir, images_filename + ".boards.hdf5")
    with h5py.File(file_path, "w") as f:
        if verbose:
            print("Create dataset (use uint8 to decrease file size)")
        total_estimate = len(piece_groups_by_id)
        image_dataset = f.create_dataset("images", (total_estimate, image_size[0], image_size[1], 3), dtype='uint8')
        if verbose:
            print("Store meta data")
        grid_config = GridConfig(30, 30, move_step=.5, prevent_overlap=True)
        grid_config.store("grid", data_dir)  # for working with state files directly
        image_dataset.attrs["grid_config.width"] = grid_config.width
        image_dataset.attrs["grid_config.height"] = grid_config.height
        image_dataset.attrs["grid_config.move_step"] = grid_config.move_step
        image_dataset.attrs["grid_config.prevent_overlap"] = grid_config.prevent_overlap
        if verbose:
            print("Go through annotations")
        plot_context = BoardPlotContext(image_size)
        counter = 0
        # Note: The group id must match the position in the h5 file
        for group_id in tqdm(range(total_estimate), position=0, leave=True):
            piece_group = piece_groups_by_id[group_id]
            if verbose:
                print("Create board with pieces")
            board = Board(grid_config, board_id=group_id)
            if not board.add_pieces_from_symbols(piece_group.pieces, max_attempts=100):
                raise Exception(f"Could not add all pieces to the board. "
                                f"Increase the attempts and try again.\n{piece_group}")
            # keep the states for later use (in the ui experiments)
            state_file.write(json.dumps(board.to_state_dict()))  # should be one line
            state_file.write("\n")  # necessary state seperator
            if verbose:
                print("Store image at associated id (this should be actually sequential)")
            image = board.to_image_array(image_size, plot_context)
            image_dataset[group_id] = image
            # Note: We cannot store the bbox crops here b.c. they result into different shapes
            # and h5py is intended to use data of the same shape. Nevertheless, cropping is only a slicing
            # operation anyway, which should not be too expensive to do on the fly.
            bboxes = [board.get_bbox(image_size[0], image_size[1], p) for p in board.pieces]
            bounding_boxes_by_group_id[group_id] = bboxes
            del image
            counter += 1
        plot_context.close()
    state_file.close()
    if verbose:
        print("Update annotations with bounding boxes")
    for annotation in annotations:
        annotation.bboxes = bounding_boxes_by_group_id[annotation.group_id]
    return annotations

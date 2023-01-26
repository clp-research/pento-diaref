import json
import os
from typing import List, Tuple

import h5py
from golmi.contrib.pentomino.objects import BoardPlotContext, Board
from golmi.server.grid import GridConfig
from tqdm import tqdm

from pentodiaref.data.generation.types import Annotation


def collect_and_store_images(target_size: Tuple[int, int], data_dir: str, split_name: str, category_name: str = None,
                             force_reindex=False, dry_run=False):
    fn = split_name
    if category_name:
        fn = f"{category_name}_{split_name}"
    annotations = Annotation.load(data_dir, fn, resolve=True)
    if dry_run:
        annotations = annotations[:100]
    if force_reindex:  # this might be necessary, when we split a larger set of samples
        print("Force re-index of annotations to be aligned with h5py index")
        for idx, a in enumerate(annotations):
            a.anno_id = idx
    annos_with_bboxes = collect_images_and_bboxes(annotations, data_dir, fn, target_size)
    if dry_run:
        return
    Annotation.store(annos_with_bboxes, fn, data_dir)


def collect_images_and_bboxes(annotations: List[Annotation], data_dir: str, filename: str, image_size, verbose=False):
    # We directly store into the state file line by line to save memory and compute
    state_file_path = os.path.join(data_dir, filename + ".states")
    if os.path.exists(state_file_path):
        os.remove(state_file_path)
    state_file = open(state_file_path, "a")

    file_path = os.path.join(data_dir, filename + ".boards.hdf5")
    with h5py.File(file_path, "w") as f:
        if verbose:
            print("Create dataset (use uint8 to decrease file size)")
        total_estimate = len(annotations)
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
        for annotation in tqdm(annotations, position=0, leave=True):
            if verbose:
                print("Create board with pieces")
            board = Board(grid_config, board_id=annotation.anno_id)
            if not board.add_pieces_from_symbols(annotation.group.pieces, max_attempts=100):
                raise Exception(f"Could not add all pieces to the board. "
                                f"Increase the attempts and try again.\n{annotation.to_json()}")
            # keep the states for later use (in the ui experiments)
            state_file.write(json.dumps(board.to_state_dict()))  # should be one line
            state_file.write("\n")  # necessary state seperator
            if verbose:
                print("Store image at associated id (this should be actually sequential)")
            image = board.to_image_array(image_size, plot_context)
            image_dataset[annotation.anno_id] = image
            if verbose:
                print("Update annotations with bounding boxes")
            # Note: We cannot store the bbox crops here b.c. they result into different shapes
            # and h5py is intended to use data of the same shape. Nevertheless, cropping is only a slicing
            # operation anyway, which should not be too expensive to do on the fly.
            bboxes = [board.get_bbox(image_size[0], image_size[1], p) for p in board.pieces]
            annotation.bboxes = bboxes
            del image
            counter += 1
        plot_context.close()
    state_file.close()
    return annotations

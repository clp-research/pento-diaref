import argparse
import os.path
import random
import numpy as np

from golmi.contrib.pentomino.symbolic.types import Shapes, Colors, RelPositions

from pentodiaref.data.generation.didactic import collect_and_store_annotations

from pentodiaref.data.generation.holdouts import create_color_holdout
from pentodiaref.data.generation.holdouts import create_position_holdout
from pentodiaref.data.generation.holdouts import create_utterance_type_holdout

from pentodiaref.data.generation.checks import check_color_split_counts
from pentodiaref.data.generation.checks import check_pos_split_counts
from pentodiaref.data.generation.checks import check_piece_colors_per_split
from pentodiaref.data.generation.checks import check_piece_pos_per_split
from pentodiaref.data.generation.checks import check_ut_split_counts
from pentodiaref.data.generation.checks import check_piece_utterances_per_split

from pentodiaref.data.generation.expressions import utterance_types_csp
from pentodiaref.data.generation.utils import load_sent_types


def cli(data_dir: str,
        train_num_sets_per_utterance_type: int,
        test_num_sets_per_utterance_type: int,
        gid_start: int = 0,
        verbose: bool = False):
    print("train_num_sets_per_utterance_type:", train_num_sets_per_utterance_type)
    print("test_num_sets_per_utterance_type:", test_num_sets_per_utterance_type)
    # 1. Prepare holdout piece configs for val and test
    shapes = list(Shapes)
    colors = list(Colors)
    positions = list(RelPositions)
    sent_types = load_sent_types(data_dir)

    # 1.(a)
    # Colors disentangled: For each shape           in train, we hold-out 2 colors (with all their positions)
    # These are 2 different colors for each shape, so that the colors have the same amount of representations
    print("Create holdout colors")
    configs_by_color_split = create_color_holdout(shapes, colors, positions, verbose=False,
                                                  color_holdout=1)
    check_color_split_counts(configs_by_color_split, len(positions))
    if verbose:
        check_piece_colors_per_split(configs_by_color_split)

    # 1.(b)
    # Position invariance: For each shape and color in train, we hold-out 1 positions
    print("Create holdout positions")
    configs_by_pos_split = create_position_holdout(configs_by_color_split["train"],
                                                   num_positions=len(positions), pos_holdout=1,
                                                   verbose=False)
    check_pos_split_counts(configs_by_pos_split)
    if verbose:
        check_piece_pos_per_split(configs_by_pos_split)

    # 1.(c)
    # Utterance type invariance: For each piece config (shape,color,pos) in train, we hold-out 1 utterance type

    # Actually, the returned splits are dicts with tuple keys: (shape,color)
    # referring to a dict with a list of "utterances" and a list of (position)
    # We have to re-create the piece configs from this information later
    print("Create holdout utterance-types")
    configs_by_uts_split = create_utterance_type_holdout(configs_by_pos_split["train"],
                                                         list(utterance_types_csp.keys()),
                                                         verbose=False)
    check_ut_split_counts(configs_by_uts_split)
    if verbose:
        check_piece_utterances_per_split(configs_by_uts_split)

    # 1.(d)
    configs_by_split = {
        "ho-color": configs_by_color_split,
        "ho-pos": configs_by_pos_split,
        "ho-uts": configs_by_uts_split,
    }

    """
    Note that the counts are still, very high. These are not "sensibly producable" 
    even when we disallow more than 2 pieces per position. 
    We check for duplicates though (which are still very unlikely to happen).
    """

    # minimum is 4 b.c. (color, shape, position) requires 3 additional pieces and if we just
    # change the minimum for that case, then the model might get a hint from the numbers
    pieces_per_set = (4, 10)
    test_split_size = 10_000
    num_sets_per_utterance_type = train_num_sets_per_utterance_type

    # We actually build the training data from these "remaining" piece configurations
    category_name = "ho-uts"

    print(f"Generate annotations for training 'data'")
    num_extra_targets = 3  # we sample additional pieces as targets for the same board
    gid_start = collect_and_store_annotations(data_dir, configs_by_split, category_name, split_name="train",
                                              num_sets_per_utterance_type=num_sets_per_utterance_type,
                                              num_extra_targets=num_extra_targets,
                                              pieces_per_set=pieces_per_set,
                                              gid_start=gid_start,
                                              sent_types=sent_types,
                                              test_split_size=test_split_size,
                                              filename="data")

    print(f"Generate annotations for evaluation '{category_name}'")
    num_extra_targets = 0  # no extra targets here as we evaluate on a holdout
    num_sets_per_utterance_type = test_num_sets_per_utterance_type

    gid_start = collect_and_store_annotations(data_dir, configs_by_split, category_name, split_name="val",
                                              num_sets_per_utterance_type=num_sets_per_utterance_type,
                                              num_extra_targets=num_extra_targets,
                                              pieces_per_set=pieces_per_set,
                                              gid_start=gid_start,
                                              sent_types=sent_types)
    gid_start = collect_and_store_annotations(data_dir, configs_by_split, category_name, split_name="test",
                                              num_sets_per_utterance_type=num_sets_per_utterance_type,
                                              num_extra_targets=num_extra_targets,
                                              pieces_per_set=pieces_per_set,
                                              gid_start=gid_start,
                                              sent_types=sent_types)

    for category_name in ["ho-color", "ho-pos"]:
        print(f"Generate annotations for evaluation '{category_name}'")
        gid_start = collect_and_store_annotations(data_dir, configs_by_split, category_name, split_name="val",
                                                  num_sets_per_utterance_type=num_sets_per_utterance_type,
                                                  num_extra_targets=num_extra_targets,
                                                  pieces_per_set=pieces_per_set,
                                                  gid_start=gid_start,
                                                  sent_types=sent_types,
                                                  mixin=list(configs_by_split[category_name]["train"]))
        gid_start = collect_and_store_annotations(data_dir, configs_by_split, category_name, split_name="test",
                                                  num_sets_per_utterance_type=num_sets_per_utterance_type,
                                                  num_extra_targets=num_extra_targets,
                                                  pieces_per_set=pieces_per_set,
                                                  gid_start=gid_start,
                                                  sent_types=sent_types,
                                                  mixin=list(configs_by_split[category_name]["train"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--data_dir', type=str,
                        default="/data/pento_diaref/didact", help="Default: /data/pento_diaref/didact")
    parser.add_argument("-n", '--train_num_sets_per_utterance_type', type=int, required=True,
                        help="The number of training boards per utterance type (5)")
    parser.add_argument("-t", '--test_num_sets_per_utterance_type', type=int, required=True,
                        help="The number of testing boards per utterance type (1)")
    parser.add_argument("-g", '--gid_start', type=int, default=0,
                        help="Offset for the global_id of the annotations")
    parser.add_argument("-v", '--verbose', action="store_true")
    parser.add_argument("-S", '--seed', type=int, default=None)
    args = parser.parse_args()
    if not os.path.exists(args.data_dir):
        raise Exception("data_dir does not exist. Please create it and try again.")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    cli(args.data_dir,
        args.train_num_sets_per_utterance_type,
        args.test_num_sets_per_utterance_type,
        args.gid_start,
        args.verbose)

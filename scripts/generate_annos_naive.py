import itertools
import os
import argparse as argparse
from golmi.contrib.pentomino.symbolic.types import Shapes, Colors, RelPositions, SymbolicPiece

from pentodiaref.data.generation.checks import check_color_split_counts, check_pos_split_counts
from pentodiaref.data.generation.naive import collect_and_store_annotations_random
from pentodiaref.data.generation.utils import load_sent_types

from pentodiaref.data.generation.holdouts import create_color_holdout
from pentodiaref.data.generation.holdouts import create_position_holdout


def cli(data_dir: str, with_ho: bool, gid_start: int, verbose: bool = False):
    shapes = list(Shapes)
    colors = list(Colors)
    positions = list(RelPositions)

    print("data_dir:", data_dir)

    # Simply allow all possible combinations
    piece_configs = [SymbolicPiece(color, shape, pos)
                     for (color, shape, pos) in itertools.product(colors, shapes, positions)]
    print("Possible (shape, color, pos) combinations:", len(piece_configs))

    if with_ho:
        print("Create holdout colors")
        configs_by_color_split = create_color_holdout(shapes, colors, positions, verbose=False,
                                                      color_holdout=1)
        check_color_split_counts(configs_by_color_split, len(positions))
        print("Create holdout positions")
        configs_by_pos_split = create_position_holdout(configs_by_color_split["train"],
                                                       num_positions=len(positions), pos_holdout=1,
                                                       verbose=False)
        check_pos_split_counts(configs_by_pos_split)
        print("Use remaining piece configs for dataset generation")
        piece_configs = configs_by_pos_split["train"]

    sent_types = load_sent_types(data_dir)

    # minimum is 4 b.c. (color, shape, position) requires 3 additional pieces and if we just
    # change the minimum for that case, then the model might get a hint from the numbers
    pieces_per_set = (4, 10)
    pieces_per_pos = 2
    targets_per_set = 4  # this means for "small" boards all are selected as targets

    # this results into 168_000 boards when we choose 4 targets per board
    # then its the same amount as we have with utos sampling (148K,10K,10K) (train,val,tes)
    number_of_sets = 42_000
    test_split_size = 10_000

    print(f"Generate annotations for 'data' training and evaluation")
    collect_and_store_annotations_random(data_dir, piece_configs,
                                         pieces_per_set, pieces_per_pos,
                                         targets_per_set, number_of_sets,
                                         test_split_size, sent_types,
                                         gid_start, verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--data_dir', type=str,
                        default="/data/pento_diaref/naive", help="Default: /data/pento_diaref/naive")
    parser.add_argument("-g", '--gid_start', type=int, default=1_000_000,
                        help="Offset for the global_id of the annotations. Default: 1_000_000")
    parser.add_argument("-ho", '--with_ho', action="store_true",
                        help="Whether to calculate a holdout from the possible target piece symbols.")
    parser.add_argument("-v", '--verbose', action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.data_dir):
        raise Exception("data_dir does not exist. Please create it and try again.")
    cli(args.data_dir,
        args.with_ho,
        args.gid_start,
        args.verbose)

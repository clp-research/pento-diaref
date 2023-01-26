import itertools
from collections import defaultdict
import random
from typing import List, Set

from golmi.contrib.pentomino.symbolic.types import Shapes, Colors, RelPositions, SymbolicPiece

""" Color holdout """


def create_color_holdout(shapes: List[Shapes], colors: List[Colors], positions: List[RelPositions],
                         color_holdout=2, verbose=False):
    configs_by_split = {
        "train": set(),
        "test": set(),
        "val": set()
    }
    for shape_idx, shape in enumerate(shapes):
        splits = __create_rotated_color_splits(shape_idx, num_colors=len(colors), num_positions=len(positions),
                                               color_holdout=color_holdout)
        if verbose:
            print(splits)
        for split, (color, position) in zip(splits, itertools.product(colors, positions)):
            piece_config = SymbolicPiece(color=color, shape=shape, rel_position=position)
            configs_by_split[split].add(piece_config)
    return configs_by_split


def __create_rotated_color_splits(shape_idx, num_colors, num_positions, color_holdout=1):
    splits = ["train"] * (num_colors * num_positions)
    pointer = shape_idx * num_positions  # for each shape, start "one whole" color later
    for _ in range(color_holdout * num_positions):  # hold out "the whole" color (all positions)
        splits[pointer % (num_colors * num_positions)] = "test"
        pointer += 1
    for _ in range(color_holdout * num_positions):
        splits[pointer % (num_colors * num_positions)] = "val"
        pointer += 1
    return splits


""" Position holdout """


def create_position_holdout(pieces: Set[SymbolicPiece], num_positions, pos_holdout=2, verbose=False):
    configs_by_split = {
        "train": set(),
        "test": set(),
        "val": set()
    }
    # We expect pieces to be a sorted set. Pieces are grouped by shape and color.
    # The positions are always in the same order.
    pieces = sorted(list(pieces))
    splits = __create_rotated_pos_splits(len(pieces), num_positions, pos_holdout)
    for split, piece_config in zip(splits, pieces):
        if verbose:
            print(piece_config, split)
        configs_by_split[split].add(piece_config)
    return configs_by_split


def __create_rotated_pos_splits(num_pieces, num_positions, pos_holdout=1, verbose=False):
    splits = ["train"] * num_pieces

    # Prepare "circle" of pos to split mappings
    pos_split = dict([(pos, "train") for pos in range(num_positions)])
    pos_pointer = 0
    for pos in range(pos_holdout):
        pos_split[pos_pointer] = "test"
        pos_pointer += 1
    for pos in range(pos_holdout):
        pos_split[pos_pointer] = "val"
        pos_pointer += 1

    if verbose:
        print("Pos-Circle", pos_split)

    rounds = -1  # we increase on "full round" at the beginning
    for idx in range(num_pieces):
        if idx % num_positions == 0:  # full round
            rounds += 1
            if verbose:
                print("Round", rounds)
        pos_pointer = (idx + rounds) % num_positions  # shift each round
        splits[idx] = pos_split[pos_pointer]
        if verbose:
            print(idx, pos_pointer, pos_split[pos_pointer])
    return splits


""" Utterance type holdout """


def create_utterance_type_holdout(pieces: Set[SymbolicPiece], utterance_types: List[int],
                                  type_holdout=1, verbose=False):
    configs_by_split = {
        "train": defaultdict(lambda: {"utterances": []}),
        "test": defaultdict(lambda: {"utterances": []}),
        "val": defaultdict(lambda: {"utterances": []})
    }
    if verbose:
        print("Pieces (select=10):", random.sample(list(pieces), 10))
    # Group positions by (shape,color) so that we can apply an utterance type to all positions
    grouped = defaultdict(list)
    for piece in pieces:
        grouped[(piece.shape, piece.color)].append(piece.rel_position)
    if verbose:
        print("Grouped:", [random.choice(list(grouped.keys())) for _ in range(5)])
    # We expect pieces to be a sorted set. Pieces are grouped by shape and color.
    # The positions are always in the same order.
    pieces = sorted(list(grouped.keys()))
    if verbose:
        print("Sorted (select=10):", random.sample(list(pieces), 10))
    splits = __create_rotated_pos_splits(len(pieces) * len(utterance_types), len(utterance_types), type_holdout)
    if verbose:
        print("Splits:", len(splits))
        print("Piece-Uts:", len(list(itertools.product(pieces, utterance_types))))
    for split, (piece_tup, ut) in zip(splits, itertools.product(pieces, utterance_types)):
        if verbose:
            print(split, (piece_tup, ut))
        configs_by_split[split][piece_tup]["utterances"].append(ut)
    # re-attach the position information
    for split in configs_by_split:
        for piece_tup in configs_by_split[split]:
            configs_by_split[split][piece_tup]["positions"] = grouped[piece_tup]
    return configs_by_split

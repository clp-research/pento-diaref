import collections
from typing import List, Set, Dict
from collections import defaultdict

from golmi.contrib.pentomino.symbolic.types import SymbolicPiece, SymbolicPieceGroup


def check_color_split_counts(configs_by_split: Dict, num_positions):
    for k, v in configs_by_split.items():
        print("Split", k, "count(s,c,p):", len(v), "count(s,c):", int(len(v) / num_positions))


def check_piece_colors_per_split(configs_by_color_split: Dict):
    for split_name, pieces in configs_by_color_split.items():
        print("Split:", split_name)
        __check_piece_colors(pieces)
        print()


def __check_piece_colors(pieces: Set):  # ignore position
    n_pieces = defaultdict(set)
    for p in pieces:
        n_pieces[p.shape].add(p.color)
    keys = sorted(list(n_pieces.keys()))
    n_colors = set()
    for k in keys:
        v = n_pieces[k]
        print(f"Piece ({len(v)}):", k, sorted(list(v)))
        n_colors.update(v)
    print(f"Colors ({len(n_colors)})", n_colors)


def check_pos_split_counts(configs_by_split: Dict):
    for k, v in configs_by_split.items():
        print("Split", k, "count(s,c,p):", len(v))


def check_piece_pos_per_split(configs_by_pos_split: Dict):
    for split_name, pieces in configs_by_pos_split.items():
        print("Split:", split_name)
        __check_piece_pos(pieces)
        print()


def __check_piece_pos(pieces: Set[SymbolicPiece]):
    n_pieces = defaultdict(set)
    for p in pieces:
        n_pieces[(p.shape, p.color)].add(p.rel_position)
    keys = sorted(list(n_pieces.keys()))
    n_pos = set()
    for k in keys:
        v = n_pieces[k]
        print(f"Piece ({len(list(v))}:", k, sorted(list(v)))
        n_pos.update(v)
    print(f"Positions ({len(n_pos)})", n_pos)


def check_ut_split_counts(configs_by_split: Dict):
    for k, v in configs_by_split.items():
        print("Split", k, "count(s,c):", len(v))


def __check_piece_utterances(pieces: Dict[tuple, Dict[str, List]]):
    keys = sorted(list(pieces.keys()))
    n_uts = set()
    for k in keys:
        uts = sorted(pieces[k]["utterances"])
        print(f"Piece ({len(uts)}):", k, uts)
        n_uts.update(uts)
    print(f"Utterance Types ({len(n_uts)})", n_uts)


def check_piece_utterances_per_split(configs_by_ut_split: Dict):
    for split_name, pieces in configs_by_ut_split.items():
        print("Split:", split_name)
        __check_piece_utterances(pieces)
        print()


def check_position_restriction(piece_sets: List[SymbolicPieceGroup]) -> Set[SymbolicPieceGroup]:
    errors = set()
    for piece_set in piece_sets:
        pos_counts = collections.defaultdict(int)
        for piece in piece_set:
            pos_counts[piece.rel_position] += 1
            if pos_counts[piece.rel_position] > 2:
                errors.add(piece_set)
    return errors


def order_by_enum(e):
    """ Return a dict of enum to idx """
    return dict([(v, idx) for (idx, v) in zip(range(len(list(e))), list(e))])


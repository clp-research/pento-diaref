from collections import defaultdict, Counter
from typing import List

from golmi.contrib.pentomino.symbolic.types import Colors, Shapes, RelPositions
from pentodiaref.data.generation.types import Annotation


def order_by_enum(e):
    """ Return a dict of enum to idx """
    return dict([(v, idx) for (idx, v) in zip(range(len(list(e))), list(e))])


def count_target_piece_colors(annotations: List[Annotation]):
    """
        Count for each shape with what colors it occurs.
    """
    n_shapes = defaultdict(Counter)
    for annotation in annotations:
        target_idx = annotation.target_idx
        target = annotation.group[target_idx]
        n_shapes[target.shape].update([target.color])
    # print counts
    order_by_shape = order_by_enum(Shapes)
    keys = sorted(list(n_shapes.keys()), key=lambda x: order_by_shape[x])  # sort by order
    n_colors = set()
    order_by_color = order_by_enum(Colors)
    for k in keys:
        v = n_shapes[k]
        print(f"Shape ({len(v)}):", k, sorted(list(v.keys()), key=lambda x: order_by_color[x]))  # sort by order
        n_colors.update(v)
    print(f"Colors ({len(n_colors)})", sorted(list(n_colors), key=lambda x: order_by_color[x]))  # sort by order
    return n_shapes


def count_target_piece_positions(annotations: List[Annotation]):
    """
        Count for each shape-color with what position it occurs.
    """
    n_pieces = defaultdict(Counter)
    for annotation in annotations:
        target_idx = annotation.target_idx
        target = annotation.group[target_idx]
        n_pieces[(target.shape, target.color)].update([target.rel_position])
    # print counts
    order_by_shape = order_by_enum(Shapes)
    order_by_color = order_by_enum(Colors)
    # sort by order shapes first, then colors
    keys = sorted(list(n_pieces.keys()), key=lambda x: (order_by_shape[x[0]], order_by_color[x[1]]))
    n_pos = set()
    order_by_pos = order_by_enum(RelPositions)
    for k in keys:
        v = n_pieces[k]
        print(f"Piece ({len(list(v))}:", k, sorted(list(v.keys()), key=lambda x: order_by_pos[x]))  # sort by order
        n_pos.update(v)
    print(f"Positions ({len(n_pos)})", sorted(list(n_pos), key=lambda x: order_by_pos[x]))  # sort by order
    return n_pieces


def count_target_piece_utterances_types(annotations: List[Annotation]):
    """
        Count for each shape-color-position with what utterance type it occurs.
    """
    n_pieces = defaultdict(Counter)  # the overall counts
    n_intended = defaultdict(Counter)  # the intended counts, where target_idx = 0
    for annotation in annotations:
        target_idx = annotation.target_idx
        target = annotation.group[target_idx]
        n_pieces[target].update([annotation.refs[0].utterance_type])
        if target_idx == 0:
            n_intended[target].update([annotation.refs[0].utterance_type])

    # now that we have selected all intended utterance types, we can sum up over the unintended ones
    n_count_unintended = defaultdict(int)
    for target, uts in n_pieces.items():
        intended = n_intended[target]
        for ut, count in uts.items():
            if ut not in intended:
                n_count_unintended[target] += count

    # print counts
    keys = sorted(list(n_pieces.keys()))
    n_ut = set()
    for k in keys:
        v = n_pieces[k]
        gt = n_intended[k]
        print(f"Piece ({len(list(v))}:", k, sorted(list(v.keys())))
        print(f"Piece ({len(list(v))}:", k, v.most_common(), "(by count)")
        print(f"Piece ({len(list(gt))}:", k, sorted(list(gt.keys())), "(intended)")
        print(f"Mismatch: {n_count_unintended[k]}")
        n_ut.update(v.keys())
    print("Mismatch overall", sum([v for v in n_count_unintended.values()]))
    print(f"Uts ({len(n_ut)})", sorted(list(n_ut)))
    return n_pieces

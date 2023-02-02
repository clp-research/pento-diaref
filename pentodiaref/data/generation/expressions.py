from typing import List, Dict, Union

from golmi.contrib.pentomino.symbolic.algos import PentoIncrementalAlgorithm
from golmi.contrib.pentomino.symbolic.types import SymbolicPieceGroup, SymbolicPiece, PropertyNames, Colors, Shapes, \
    RelPositions

from pentodiaref.data.generation.types import Reference, Annotation

pia_csp = PentoIncrementalAlgorithm([PropertyNames.COLOR, PropertyNames.SHAPE, PropertyNames.REL_POSITION],
                                    start_tokens=["Take"])

# based on preference order [COLOR, SHAPE, REL_POSITION]
utterance_types_csp = {
    0: [PropertyNames.COLOR],
    1: [PropertyNames.COLOR, PropertyNames.SHAPE],
    2: [PropertyNames.COLOR, PropertyNames.REL_POSITION],
    3: [PropertyNames.COLOR, PropertyNames.SHAPE, PropertyNames.REL_POSITION],
    4: [PropertyNames.SHAPE],
    5: [PropertyNames.SHAPE, PropertyNames.REL_POSITION],
    6: [PropertyNames.REL_POSITION],
}


def properties_to_utterance_type(property_values: Dict[PropertyNames, Union[Colors, Shapes, RelPositions]]):
    pvs = tuple(sorted([k for k in property_values]))
    prop2types = dict([(tuple(sorted(v)), k) for k, v in utterance_types_csp.items()])
    utt = prop2types[pvs]
    if utt is None:
        raise Exception(f"{pvs} not in {prop2types}")
    return utt


def __collect_ref(pia, pieces: SymbolicPieceGroup, target: SymbolicPiece, sent_types: Dict[str, Dict] = None):
    expression, property_values, is_discriminating = pia.generate(pieces, target,
                                                                  is_selection_in_pieces=True,
                                                                  return_expression=True)
    if not is_discriminating:
        print("Warn: Target description is ambiguous")
        print("Pieces:", pieces)
        print("Target:", target)

    # determine utterance type (although we create a board for a
    # specific utterance type, it may be different e.g. for humans)
    ut_idx = properties_to_utterance_type(property_values)

    # determine the sentence type (the type of the actual utterance)
    sent_type = -1
    if sent_types is not None:
        sent_type = sent_types["stoi"][expression.lower()]
    return Reference("ia", ut_idx, sent_type, expression, property_values)


def generate_referring_expression(sample_idx: int, group_idx: int, piece_group: SymbolicPieceGroup, target_idx: int,
                                  pias: List[PentoIncrementalAlgorithm],
                                  sent_types: Dict[str, Dict] = None):
    target_piece = piece_group[target_idx]
    refs = []
    for pia in pias:
        ref = __collect_ref(pia, piece_group, target_piece, sent_types)
        refs.append(ref)
    return Annotation(sample_idx, group_idx, target_idx, piece_group, refs)

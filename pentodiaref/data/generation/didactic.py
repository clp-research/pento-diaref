import random
from collections import defaultdict
from typing import Tuple, Dict, Iterable, Counter, List

from golmi.contrib.pentomino.symbolic.sampling import UtteranceTypeOrientedDistractorSetSampler
from golmi.contrib.pentomino.symbolic.types import SymbolicPieceGroup, SymbolicPiece, Rotations
from tqdm import tqdm

from pentodiaref.data.generation.checks import check_position_restriction
from pentodiaref.data.generation.expressions import utterance_types_csp, pia_csp, generate_referring_expression
from pentodiaref.data.generation.types import Annotation
from pentodiaref.data.generation.utils import add_gid_and_split_name


def __select_annotations_with_unintended_utterances_types(annotations: List[Annotation]) -> List[Annotation]:
    """
        Count for each shape-color-position with what utterance type it occurs.
    """
    n_intended = defaultdict(Counter)  # the intended counts, where target_idx = 0
    for annotation in annotations:
        target_idx = annotation.target_idx
        target = annotation.group[target_idx]
        ut = annotation.refs[0].utterance_type
        if target_idx == 0:
            n_intended[target].update([ut])

    # now that we have selected all intended utterance types
    n_unintended = list()
    for annotation in annotations:
        target_idx = annotation.target_idx
        target = annotation.group[target_idx]
        ut = annotation.refs[0].utterance_type
        intended = n_intended[target]
        if ut not in intended:
            n_unintended.append(annotation)
    return n_unintended


def __filter_unintended_samples(samples):
    """
    For utterance-type oriented sampling we choose extra targets randomly, so that a board is not dedicated to a
    single target. But doing so results sometimes into utterance types that were not intended for the target to
    be present in the training data. Thus we need to mark these cases, so that we can avoid them during training.
    """
    unintended = __select_annotations_with_unintended_utterances_types(samples)
    print(f"Found {len(unintended)} unintended samples to be removed from initially {len(samples)} ones.")
    unintended_ids = [a.anno_id for a in unintended]
    samples = [a for a in samples if a.anno_id not in unintended_ids]
    return samples


def convert_piece_configs_to_uts_dicts(pieces: Iterable[SymbolicPiece]):
    # Group positions by (shape,color) so that we can apply an utterance type to all positions
    uts_dicts = defaultdict(lambda: {"positions": []})
    for piece in pieces:
        piece_tup = (piece.shape, piece.color)
        uts_dicts[piece_tup]["positions"].append(piece.rel_position)
    for piece_tup in uts_dicts:
        uts_dicts[piece_tup]["utterances"] = list(utterance_types_csp.keys())
    return uts_dicts


def collect_and_store_annotations(data_dir, splits, category_name, split_name,
                                  num_sets_per_utterance_type: int,
                                  num_extra_targets: int,
                                  pieces_per_set: Tuple[int, int],
                                  gid_start: int,
                                  sent_types: Dict[str, Dict],
                                  filename=None, mixin=[],
                                  test_split_size: int = None,
                                  verbose=False):
    sets_per_utterance_type = dict((t, num_sets_per_utterance_type) for t in range(7))

    category_splits = splits[category_name]
    uts_dicts = category_splits[split_name]

    if not isinstance(uts_dicts, dict):  # convert to handle all splits in the same way
        print("Convert PieceConfigs {} to Uts_Dicts".format(uts_dicts.__class__))
        uts_dicts = convert_piece_configs_to_uts_dicts(uts_dicts)

    samples, piece_sets = collect_samples_utterance_oriented(uts_dicts,
                                                             pieces_per_set,
                                                             sets_per_utterance_type,
                                                             sent_types=sent_types,
                                                             non_targets=mixin,
                                                             num_extra_targets=num_extra_targets)
    gid = gid_start
    fn = f"{category_name}_{split_name}"
    if filename:
        fn = filename

    if filename == "data":  # training data
        if test_split_size is None:
            test_split_size = int(len(samples) * .1)  # 10 percent test splits size
        te, ts = len(samples) - 2 * test_split_size, test_split_size
        train, val, test = samples[:te], samples[te:te + ts], samples[te + ts:]

        if num_extra_targets > 0:
            # since we are doing this only here, this means that test could include unseen utterance types
            train = __filter_unintended_samples(train)

        for split_name, split_samples in zip(["train", "val", "test"], [train, val, test]):
            gid = add_gid_and_split_name(split_samples, gid, fn + "_" + split_name)
            Annotation.store(split_samples, fn + "_" + split_name, data_dir)
    else:  # holdouts
        if num_extra_targets > 0:
            samples = __filter_unintended_samples(samples)
        gid = add_gid_and_split_name(samples, gid, fn)

        if verbose:
            print("==== Annotation Sample ====")
            print(random.choice(samples).to_json())
            print("===========================")

        Annotation.store(samples, fn, data_dir)
    print()
    return gid


def collect_samples_utterance_oriented(uts_dicts: Dict,
                                       pieces_per_set: Tuple,
                                       sets_per_utterance_type: Dict,
                                       pieces_per_pos: int = 2,
                                       n_retries: int = 100,
                                       non_targets: List[SymbolicPiece] = [],
                                       num_extra_targets: int = 0,
                                       sent_types: Dict[str, Dict] = None,
                                       verbose=False):
    """
        uts_dicts: A dict with tuple (shape,color) keys pointing to a dict with "positions" and "utterances"
                    {(F, blue): {'positions': [center, center], 'utterances': [0, 1, 2, 3, 4, 5, 6]}}
    """
    assert pieces_per_set[0] >= 4, "(color, shape, position) requires at least 3 additional pieces"
    if verbose:
        for ut, count in sets_per_utterance_type.items():
            print("Sets per Type:", ut, "->", count)

    # build list of possible pieces (to sample from)
    pieces = []
    utterances_count = 0
    for piece_tup, data in uts_dicts.items():
        assert "positions" in data, f"{piece_tup}: {data}"  # we use a defaultdict, which can get ugly
        for position in data["positions"]:
            pieces.append(SymbolicPiece(color=piece_tup[1], shape=piece_tup[0], rel_position=position))
        for ut in data["utterances"]:
            utterances_count += (len(data["positions"]) * sets_per_utterance_type[ut])

    if num_extra_targets > 0:  # when we want to have multiple targets per board (but these are randomly chosen)
        utterances_count *= (1 + num_extra_targets)

    # uts_dict based sampling to allow for specific uts only
    pos_errors = set()
    samples = []
    piece_groups = []
    total_estimate = int(utterances_count)
    print("Piece Configs:", len(pieces))
    print("Mix-in Non-Targets:", len(non_targets))
    if non_targets:
        pieces += non_targets
    print("Utterances:", utterances_count)
    print("Pieces per Set:", pieces_per_set)
    print("Estimated number of samples", total_estimate)
    # CLEVR produces 1,000,000 questions (here: ref. exps.)
    counter_loop = tqdm(range(total_estimate), position=0, leave=True)
    counter = 0
    for piece_tup, uts_dict in uts_dicts.items():
        for position in uts_dict["positions"]:
            target_piece = SymbolicPiece(color=piece_tup[1], shape=piece_tup[0], rel_position=position)
            target_piece.rotation = Rotations.from_random()  # apply any rotation, this is only for visual effect
            sampler = UtteranceTypeOrientedDistractorSetSampler(pieces, target_piece, n_retries)
            for ut_idx in uts_dict["utterances"]:
                utterance_type = utterance_types_csp[ut_idx]
                utterance_number_of_sets = sets_per_utterance_type[ut_idx]
                distractors_groups = sampler.sample_many_distractor_groups(utterance_type, utterance_number_of_sets,
                                                                           pieces_per_set, rotate_pieces=True)
                for distractor_group in distractors_groups:
                    # we use list here to retain order: target is identified by the list idx
                    piece_group = SymbolicPieceGroup([target_piece] + distractor_group.pieces)
                    piece_groups.append(piece_group)
                    target_indices = [0]  # first piece is always initial target
                    if num_extra_targets > 0:
                        target_indices += random.sample(list(range(len(piece_group)))[1:], num_extra_targets)
                    for target_idx in target_indices:
                        sample = generate_referring_expression(counter, piece_group, target_idx, [pia_csp], sent_types)
                        samples.append(sample)
                        counter += 1  # starting the ids with 0 for easier reference of the images in the h5py
                        # check that utterance-type matches ia prediction
                        # Note: we can only do this for the pieces for which the board was created initially
                        if target_idx == 0:
                            ia_props = [pn for pn in sample.refs[0].property_values]  # collect keys
                            assert set(ia_props) == set(utterance_type), \
                                f"ia_props: {set(ia_props)} != utterance_type: {set(utterance_type)} \n" \
                                f"target_piece: {target_piece}\n" \
                                f"piece_list: {piece_group}\n"
                        counter_loop.update()
                        if counter % 100 == 0:
                            counter_loop.refresh()
                pos_errors.update(check_position_restriction(distractors_groups))
    counter_loop.refresh()
    print("Position exceptions:", len(pos_errors))
    if verbose:
        if len(pos_errors) > 0:
            for error in pos_errors:
                print(error)
    if len(piece_groups) != len(set(piece_groups)):
        print("Warn: There are duplicates, but this is fine, when for different targets.")
        print("Groups:", len(set(piece_groups)), "!=", len(piece_groups))
    else:
        print("No duplicates detected")
    return samples, piece_groups

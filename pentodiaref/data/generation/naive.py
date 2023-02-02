import random
from typing import List, Tuple, Dict

from golmi.contrib.pentomino.symbolic.algos import PentoIncrementalAlgorithm
from golmi.contrib.pentomino.symbolic.sampling import RestrictivePieceConfigGroupSampler
from golmi.contrib.pentomino.symbolic.types import SymbolicPiece
from tqdm import tqdm

from pentodiaref.data.generation.expressions import generate_referring_expression, pia_csp
from pentodiaref.data.generation.types import Annotation
from pentodiaref.data.generation.utils import add_gid_and_split_name


def collect_samples_random(sampler: RestrictivePieceConfigGroupSampler, pias: List[PentoIncrementalAlgorithm],
                           targets_per_set=3, pieces_per_set=(4, 10), n_sets=100,
                           sent_types: Dict[str, Dict] = None):
    samples = []
    piece_groups = []
    if targets_per_set:
        total_estimate = n_sets * targets_per_set
    else:  # take all as targets
        total_estimate = int(n_sets * (pieces_per_set[0] + pieces_per_set[1]) / 2)
    print("Estimated number of samples", total_estimate, "(should vary around this)")
    # CLEVR produces 1,000,000 questions (here: ref. exps.)
    counter_loop = tqdm(range(total_estimate), position=0, leave=True)
    group_idx_counter = 0
    idx_counter = 0
    for sample_idx in range(n_sets):
        # duplicate sets actually almost never happen (not seen for 100K sets)
        set_size = random.randint(pieces_per_set[0], pieces_per_set[1])
        piece_group = sampler.sample_with_position_restriction(set_size)
        piece_groups.append(piece_group)
        piece_idxs = list(range(len(piece_group)))
        if targets_per_set:  # if wanted, chose just a certain number of pieces to be a target piece
            piece_idxs = random.sample(piece_idxs, k=targets_per_set)
        for piece_idx in piece_idxs:
            sample = generate_referring_expression(idx_counter, group_idx_counter,
                                                   piece_group, piece_idx, pias, sent_types)
            samples.append(sample)
            idx_counter += 1
            counter_loop.update()
        if idx_counter % 100 == 0:
            counter_loop.refresh()
        group_idx_counter += 1
    counter_loop.refresh()
    if len(piece_groups) != len(set(piece_groups)):
        print("Warn: There are duplicates, but this is fine, when for different targets.")
        print("Groups:", len(set(piece_groups)), "!=", len(piece_groups))
    else:
        print("No duplicates detected")
    return samples, piece_groups


def collect_and_store_annotations_random(data_dir: str, piece_config: List[SymbolicPiece],
                                         pieces_per_set: Tuple[int, int], pieces_per_pos: int,
                                         targets_per_set: int, number_of_sets: int,
                                         test_split_size: int, sent_types: Dict[str, Dict],
                                         gid_start: int, verbose=False):
    if data_dir is None:
        raise Exception("There is no data_dir given.")

    sampler = RestrictivePieceConfigGroupSampler(piece_config, pieces_per_pos=pieces_per_pos)
    pias = [pia_csp]

    samples, piece_sets = collect_samples_random(sampler, pias, targets_per_set, pieces_per_set, number_of_sets,
                                                 sent_types)

    expected_count = number_of_sets * targets_per_set
    assert len(samples) == expected_count, f"There should be {expected_count} samples, but {len(samples)}"

    random.shuffle(samples)
    te, ts = expected_count - 2 * test_split_size, test_split_size
    train, val, test = samples[:te], samples[te:te + ts], samples[te + ts:]

    gid = gid_start
    for split_name, split_samples in zip(["train", "val", "test"], [train, val, test]):
        gid = add_gid_and_split_name(split_samples, gid, "data_" + split_name)

    if verbose:
        print("==== Annotation Sample ====")
        print(random.choice(samples).to_json())
        print("===========================")

    for split_name, split in zip(["train", "val", "test"], [train, val, test]):
        Annotation.store(split, "data_" + split_name, data_dir)
    print()
    return gid

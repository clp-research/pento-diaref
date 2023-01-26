import argparse

from pentodiaref.data.generation.types import Annotation


def get_targets(data_dir, split_name, category="data"):
    file_name = f"{category}_{split_name}"
    annos = Annotation.load(data_dir, file_name, resolve=True)
    targets = set([a.group[a.target_idx] for a in annos])
    return targets


def cli(didact_dir, naive_dir):
    for split_name in ["train", "val", "test"]:
        print("Check", split_name)
        targets_didact = get_targets(didact_dir, split_name)
        targets_naive = get_targets(naive_dir, split_name)
        print(f"Compare {len(targets_didact)} didact targets with {len(targets_naive)} naive targets")
        diff_didact = targets_didact - targets_naive
        print(f"{len(diff_didact)} targets in didact, but not in naive")
        diff_naive = targets_naive - targets_didact
        print(f"{len(diff_naive)} targets in naive, but not in didact")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--didact_dir', type=str, required=True,
                        default="/data/pento_diaref/didact", help="Default: /data/pento_diaref/didact")
    parser.add_argument("-n", '--naive_dir', type=str, required=True,
                        default="/data/pento_diaref/naive", help="Default: /data/pento_diaref/naive")
    args = parser.parse_args()
    cli(args.didact_dir, args.naive_dir)

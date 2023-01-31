import os
from argparse import ArgumentParser
from collections import OrderedDict

import torchmetrics as tm
import json

from pentodiaref.data.utils import load_annotations


def evaluate_file(results_dir, results_file, annos_by_gid, stage_name):
    """
    [{
     'global_id': 39939,
     'prediction': 'take the brown piece',
     'reference': 'take the brown piece'
     }]
    """
    print()
    with open(f"{results_dir}/{results_file}", "r") as f:
        results = json.load(f)
    print("=" * 20)
    print("File:", results_file)
    print("Results:", len(results))

    def bleu1():
        return tm.BLEUScore(n_gram=1, compute_on_step=False)

    def rouge():
        return tm.text.ROUGEScore(rouge_keys="rouge1", compute_on_step=False)

    # split_names = ["overall", "data", "ho-color", "ho-pos", "ho-uts"]
    split_names = ["data", "ho-color", "ho-pos", "ho-uts"]
    split_names = [f"{s}_{stage_name}" for s in split_names]

    bleu_scores = OrderedDict((d, bleu1()) for d in split_names)
    # rouge_scores = OrderedDict((d, rouge()) for d in split_names)
    errors = OrderedDict((d, 0) for d in split_names)
    totals = OrderedDict((d, 0) for d in split_names)

    for result in results:
        global_id = result["global_id"]
        text = result["prediction"]
        target = result["reference"]

        # we remove "easy" start tokens (that are always the same anyway)
        if text.startswith("take the ") and target.startswith("take the "):
            text = text[len("take the "):]
            target = target[len("take the "):]

        anno = annos_by_gid[global_id]
        split_name = anno["split_name"]
        totals[split_name] += 1

        if text != target:
            errors[split_name] += 1
            # errors[f"overall_{stage_name}"] += 1

        # rouge_scores[split_name].update(text, [target])
        bleu_scores[split_name].update([text], [[target]])

        # totals[f"overall_{stage_name}"] += 1
        # rouge_scores[f"overall_{stage_name}"].update(text, [target])
        # bleu_scores[f"overall_{stage_name}"].update([text], [[target]])

    for split_name in split_names:
        print(split_name)
        print("Errors:", errors[split_name], totals[split_name])
        print("SentA:", "{:.2f}".format(1 - (errors[split_name] / totals[split_name])))
        bleu_score = bleu_scores[split_name].compute()
        print("BLEU@1:", round(bleu_score.item(), 2))
        # rouge_score = rouge_scores[split_name].compute()["rouge1_fmeasure"]
        # print("ROUGE@1:", round(rouge_score.item(), 2))
        print()


def annos_by_global_id(data_dir, stage_name):
    data_categories = ["ho-pos", "ho-color", "ho-uts", "data"]
    by_gid = {}
    for category in data_categories:
        data_name = category + "_" + stage_name
        for anno in load_annotations(data_dir, data_name):
            by_gid[anno["global_id"]] = anno
    return by_gid


def main(args):
    results_dir = args.results_dir

    if args.results_file is None:
        result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")
                        and args.stage_name in f.split(".")]
    else:
        result_files = [args.results_file]
    print("Evaluate the following files:")
    for file in result_files:
        print(file)
    print()

    annos_by_gid = annos_by_global_id(args.data_dir, args.stage_name)

    result_files.sort()
    for result_file in result_files:
        evaluate_file(results_dir, result_file, annos_by_gid, args.stage_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="The folder with the annotation files.")
    parser.add_argument("--results_dir", default="results", help="Default: results")
    parser.add_argument("--results_file", help="A path to a specific prediction file to be evaluated.")
    parser.add_argument("--stage_name", required=True)
    args = parser.parse_args()
    main(args)

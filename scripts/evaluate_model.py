import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from pentodiaref.evaluation import LitModelEvaluationWrapper
from pentodiaref.models.classifier import LitClassifierModel
from pentodiaref.models.lstm import LitLstmModel
from pentodiaref.models.transformer import LitTransformerModel


def build_ckpt_path(model_dir, model_name):
    if not model_name.endswith(".ckpt"):
        model_name += ".ckpt"
    ckpt_path = os.path.join(model_dir, model_name)
    print("Load model from:", ckpt_path)
    return ckpt_path


def load_model(ckpt_path, model_name, ablation_mode=None):
    model = None
    if model_name.startswith("classifier"):
        model = LitClassifierModel.load_from_checkpoint(ckpt_path)
    if model_name.startswith("lstm"):
        model = LitLstmModel.load_from_checkpoint(ckpt_path)
    if model_name.startswith("transformer"):
        model = LitTransformerModel.load_from_checkpoint(ckpt_path)
        if ablation_mode:
            model.visual_sequence_encoding.ablation_mode = ablation_mode
    if model is None:
        raise Exception(f"Cannot handle model '{model_name}' because it is unknown.")
    return model


def main(args):
    ckpt_path = build_ckpt_path(args.model_dir, args.model_name)
    model = load_model(ckpt_path, args.model_name, args.ablation_mode)
    if not os.path.exists(args.results_dir):
        raise Exception(f"Please create a directory 'results' and try again.")
    eval_model = LitModelEvaluationWrapper(model=model, stage_name=args.stage_name, model_name=args.model_name,
                                           data_dir=args.data_dir, results_dir=args.results_dir,
                                           batch_size=args.batch_size, dry_run=args.dry_run)
    print("Detected GPU to use:", args.gpu)
    trainer = pl.Trainer(accelerator="gpu",
                         devices=[args.gpu],
                         logger=False)
    if args.stage_name == "val":
        trainer.validate(eval_model)
    else:
        trainer.test(eval_model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", default="results", help="Default: results")
    parser.add_argument("--model_dir", default="saved_models", help="Default: saved_models")
    parser.add_argument("--model_name", required=True,
                        help="[lstm,transformer,transformer-vse,classifier-vse]")
    parser.add_argument("--stage_name", type=str, required=True, help="[val,test]")
    parser.add_argument("--gpu", type=int, default=0)  # select a specific gpu
    parser.add_argument("--ablation_mode", type=str, help="[replace_random,add_random,random_types,random_regions]")
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--dry_run", default=False)
    args = parser.parse_args()
    main(args)

from argparse import ArgumentParser

import torch
from pytorch_lightning.loggers import TensorBoardLogger

from datetime import datetime
import pytorch_lightning as pl

from pentodiaref.models.lstm import LitLstmModel

torch.set_num_threads(2)


def main(args):
    # tensorboard filter: (metrics)|(loss)|(grad_2.0_norm_total_epoch)
    # tensorboard filter: train|val
    if args.gpu_fraction is not None:
        print("Set GPU fraction to:", args.gpu_fraction)
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    data_mode = args.data_mode
    project_name = data_mode.attach_name_to(args.model_name)
    print("Data mode:", data_mode.mode)
    print("Log directory:", args.logdir)
    tb_logger = TensorBoardLogger(
        save_dir=args.logdir,
        name=datetime.now().strftime(f"{project_name}/M%mD%d"),  # this is used for ModelCheckpoint path !
        version=datetime.now().strftime("%H%M%S"),  # this is used for ModelCheckpoint path !
    )
    dict_args = vars(args)
    model = LitLstmModel(**dict_args)
    print("Detected GPU to use:", args.gpu)
    trainer = pl.Trainer(accelerator="gpu",
                         devices=[args.gpu],
                         max_epochs=100,
                         logger=tb_logger,
                         val_check_interval=.1,
                         gradient_clip_val=10,
                         resume_from_checkpoint=args.resume
                         )
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--logdir", default="/cache/tensorboard-logdir")
    parser.add_argument("--model_name", type=str, default="lstm")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry_run", action='store_true')
    parser.add_argument("--gpu", type=int, default=0)  # select a specific gpu
    parser.add_argument("--gpu_fraction", type=float, default=None)
    parser = LitLstmModel.add_model_specific_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)

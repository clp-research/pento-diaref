from typing import Any, Dict
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics as tm
from torch.utils.data import DataLoader

from pentodiaref.data.loaders import PentoBoardsDataset, DataMode
from pentodiaref.data.utils import load_sent_types
from pentodiaref.models.vision.vse import VisualSequenceEncoding


class LitClassifierModel(pl.LightningModule):

    # noinspection PyUnusedLocal
    def __init__(self, model_name: str,
                 data_dir: str,
                 piecenet_arch: str,
                 pretraining: str,
                 batch_size: int,
                 data_mode=DataMode("sequential_classification"),  # that's basically the only possible one for now
                 dry_run: bool = False,
                 d_model: int = 512,
                 layer_norm_eps: float = 1e-5, **kwargs):
        super(LitClassifierModel, self).__init__()
        self.save_hyperparameters()
        if dry_run:
            print("Dry-run detected!")
        self.dry_run = dry_run
        self.data_dir = data_dir
        self.model_name = data_mode.attach_name_to(model_name)

        self.vocab = None
        self.data_mode = data_mode
        self.max_pieces = 10

        self.visual_sequence_encoding = VisualSequenceEncoding(d_model=d_model,
                                                               layer_norm_eps=layer_norm_eps,
                                                               device=self.device)

        # metrics
        self.train_acc = tm.Accuracy()
        self.val_acc = tm.Accuracy()

        # initialized during setup()
        self.sent_predictor, self.sent_types = None, None
        self.train_data, self.val_data = None, None

    def __init_sent_predictor(self):
        # can we predict the sentences already from the image embeddings
        num_classes = len(self.sent_types["stoi"])
        num_inputs = self.max_pieces * self.hparams.d_model
        self.sent_predictor = nn.Linear(in_features=num_inputs, out_features=num_classes)

    def configure_callbacks(self):
        monitor_metric = "acc/data/val"  # also adjust checkpoint filename!
        ckpt_file_name = self.model_name + "-epoch={epoch:02d}-acc={" + monitor_metric + ":.2f}"
        return [
            pl.callbacks.EarlyStopping(monitor=monitor_metric, mode="max", patience=20, verbose=True),
            pl.callbacks.ModelCheckpoint(dirpath="saved_models",  # relative to working dir
                                         filename=ckpt_file_name,
                                         monitor=monitor_metric, mode="max", save_top_k=3, verbose=True,
                                         auto_insert_metric_name=False)
        ]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitClassifierModel")
        parser.add_argument("--batch_size", type=int, default=24)
        parser.add_argument("--d_model", type=int, default=512)
        parser.add_argument("--piecenet_arch", type=str, default="piecenet34")
        parser.add_argument("--pretraining", type=str, help="[none,imagenet,checkpoint]", default="imagenet")
        parser.add_argument("--data_mode", type=DataMode, default=DataMode("sequential_classification"),
                            help="[only_targets, default_generation, sequential_generation, "
                                 "default_classification, sequential_classification]")
        return parent_parser

    def prepare_data(self):
        try:
            load_sent_types(self.data_dir)
        except Exception as e:
            print("Cannt load sent_types.json. Please prepare the file before training this model.")
            print(e)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["sent_types"] = self.sent_types  # store the sent_types to init on_load_checkpoint

    def on_load_checkpoint(self, checkpoint):
        # load the uninitialized layers
        if "sent_types" in checkpoint:
            self.sent_types = checkpoint["sent_types"]
        else:  # load from hard-drive
            self.sent_types = load_sent_types(self.data_dir)
        self.__init_sent_predictor()

    def setup(self, stage=None):
        if self.sent_types is None:  # for initial training load from hard-drive (store in checkpoint for eval)
            self.sent_types = load_sent_types(self.data_dir)

        if stage == "fit" or stage is None:
            if self.dry_run:  # use validation data alone (as it loads faster)
                self.train_data = PentoBoardsDataset(self.sent_types, "data", "val", self.data_dir, self.data_mode,
                                                     self.max_pieces)
                self.val_data = PentoBoardsDataset(self.sent_types, "data", "val", self.data_dir, self.data_mode,
                                                   self.max_pieces)
                data = self.train_data.get_data()[:1000]
                self.train_data.set_data(data[:800])
                print("Dry-run data/train:", len(self.train_data))
                self.val_data.set_data(data[800:])
                print("Dry-run data/val:", len(self.val_data))
            else:
                self.train_data = PentoBoardsDataset(self.sent_types, "data", "train", self.data_dir, self.data_mode,
                                                     self.max_pieces)
                self.val_data = PentoBoardsDataset(self.sent_types, "data", "val", self.data_dir, self.data_mode,
                                                   self.max_pieces)
        if self.sent_predictor is None:
            self.__init_sent_predictor()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01, eps=1e-08)  # defaults
        return optimizer

    def forward(self, inputs) -> Any:
        visual_token_embeddings, visual_padding = self.visual_sequence_encoding(inputs)
        batch_size = visual_padding.shape[0]
        # L x B x 512
        piece_features = visual_token_embeddings.permute(dims=[1, 0, 2])
        piece_features = piece_features.reshape((batch_size, -1))
        logits = self.sent_predictor(piece_features)
        return logits

    def translate(self, sent_idx: torch.Tensor):
        sent = self.sent_types["itos"][str(sent_idx.item())]
        return sent

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        logits = self(inputs)  # align lengths for loss computation
        loss = F.cross_entropy(logits, labels)
        self.log("loss/data/train", loss)

        predictions = torch.argmax(logits, dim=1)
        acc = self.train_acc(predictions, labels)
        self.log("acc/data/train", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        logits = self(inputs)  # align lengths for loss computation
        loss = F.cross_entropy(logits, labels)
        self.log("loss/data/val", loss)

        predictions = torch.argmax(logits, dim=1)
        acc = self.train_acc(predictions, labels)
        self.log("acc/data/val", acc)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0,
                          collate_fn=self.train_data.collate)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0,
                          collate_fn=self.val_data.collate)

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

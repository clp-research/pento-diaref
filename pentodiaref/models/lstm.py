from typing import Any, Dict
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics as tm
from torch.utils.data import DataLoader

from pentodiaref.data.loaders import PentoBoardsDataset, DataMode
from pentodiaref.data.utils import load_vocab, create_vocab, store_vocab
from pentodiaref.metrics import Bleu1Metric, TextLogger, update_metrics_with_translations, \
    reset_metrics, log_category_metrics
from pentodiaref.models.vision.resnet import LitPieceNet


class LitLstmModel(pl.LightningModule):

    # noinspection PyUnusedLocal
    def __init__(self, model_name: str, data_dir: str, data_mode: DataMode,
                 batch_size: int, lstm_hidden_size: int, word_embedding_dim: int,
                 dropout: float, lr: float, l2: float, dry_run: bool = False, **kwargs):
        super(LitLstmModel, self).__init__()
        self.save_hyperparameters()
        if dry_run:
            print("Dry-run detected!")
        self.dry_run = dry_run
        self.data_dir = data_dir
        self.model_name = data_mode.attach_name_to(model_name)

        self.data_mode = data_mode
        self.max_pieces = None
        self.max_length = 9 + 1  # 9 + <e> (ignore <s>!)

        self.target_piecenet = LitPieceNet()

        self.context_resnet = self.target_piecenet  # use the same piecenet for context
        lstm_input_dim = 2 * self.target_piecenet.image_embedding_dims + 5 + word_embedding_dim

        self.lstm_cell = nn.LSTMCell(input_size=lstm_input_dim, hidden_size=lstm_hidden_size)
        self.word_embeddings_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # validation metrics
        self.train_metrics = tm.MetricCollection({"bleu1": Bleu1Metric(),
                                                  "text": TextLogger(store_max=5)
                                                  })
        self.val_metrics = tm.MetricCollection({"bleu1": Bleu1Metric(),
                                                "text": TextLogger(store_max=5)
                                                })

        # initialized during setup()
        self.train_data, self.val_data = None, None
        self.tokenizer, self.vocab, self.pad_token, self.start_token, self.end_token = None, None, None, None, None
        self.word_embeddings, self.word_predictor = None, None

    def configure_callbacks(self):
        monitor_metric = "bleu1/data/val"  # also adjust checkpoint filename!
        filename = self.model_name + "-epoch={epoch:02d}-bleu1={" + monitor_metric + ":.2f}"
        return [
            pl.callbacks.EarlyStopping(monitor=monitor_metric, mode="max", patience=20, verbose=True),
            pl.callbacks.ModelCheckpoint(dirpath="saved_models",
                                         filename=filename,
                                         monitor=monitor_metric, mode="max", save_top_k=3, verbose=True,
                                         auto_insert_metric_name=False)
        ]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitLstmModel")
        parser.add_argument("--data_mode", type=DataMode, default=DataMode("default_generation"))
        parser.add_argument("--batch_size", type=int, default=24)
        parser.add_argument("--lstm_hidden_size", type=int, default=1024)
        parser.add_argument("--word_embedding_dim", type=int, default=512)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--l2", type=float, default=1e-4)
        return parent_parser

    def prepare_data(self):
        try:
            load_vocab(self.data_dir)
        except Exception as e:
            print("The following exception might be fine, when running this the first time")
            print(e)
            vocab = create_vocab(self.data_dir)
            store_vocab(vocab, self.data_dir)

    def close(self):
        self.train_data.close()
        for ds in self.val_data:
            ds.close()
        for ds in self.test_data:
            ds.close()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["vocab"] = self.vocab  # store the vocabulary to init on_load_checkpoint

    def on_load_checkpoint(self, checkpoint):
        # load the uninitialized layers
        if "vocab" in checkpoint:
            self.vocab = checkpoint["vocab"]
        else:  # load from hard-drive
            self.vocab = load_vocab(self.data_dir)
        self.word_embeddings = nn.Embedding(num_embeddings=(len(self.vocab)),
                                            embedding_dim=self.hparams.word_embedding_dim,
                                            padding_idx=self.vocab(["<p>"])[0])
        self.word_predictor = nn.Linear(in_features=self.hparams.lstm_hidden_size, out_features=(len(self.vocab)))

    def setup(self, stage=None):
        if self.vocab is None:  # for initial training load from hard-drive (store in checkpoint for eval)
            self.vocab = load_vocab(self.data_dir)
        self.pad_token = self.vocab(["<p>"])[0]
        self.start_token = self.vocab(["<s>"])[0]
        self.end_token = self.vocab(["<e>"])[0]

        if stage == "fit" or stage is None:
            if self.dry_run:  # use validation data alone (as it loads faster)
                self.train_data = PentoBoardsDataset(self.vocab, "data", "val", self.data_dir, self.data_mode)
                self.val_data = PentoBoardsDataset(self.vocab, "data", "val", self.data_dir, self.data_mode)
                data = self.train_data.get_data()[:10000]
                self.train_data.set_data(data[:8000])
                print("Dry-run data/train:", len(self.train_data))
                self.val_data.set_data(data[8000:])
                print("Dry-run data/val:", len(self.val_data))
            else:
                self.train_data = PentoBoardsDataset(self.vocab, "data", "train", self.data_dir, self.data_mode)
                self.val_data = PentoBoardsDataset(self.vocab, "data", "val", self.data_dir, self.data_mode)
        if self.word_embeddings is None:
            self.word_embeddings = nn.Embedding(num_embeddings=len(self.vocab),
                                                embedding_dim=self.hparams.word_embedding_dim,
                                                padding_idx=self.pad_token)
        if self.word_predictor is None:
            self.word_predictor = nn.Linear(in_features=self.hparams.lstm_hidden_size, out_features=len(self.vocab))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer]  # , [scheduler]

    def forward(self, inputs, utterances=None, max_length=None) -> Any:
        target_images = inputs["target_image"]
        context_images = inputs["context_images"]
        target_attributes = inputs["target_attributes"]

        target_embeddings = self.target_piecenet(target_images)
        target_embeddings = target_embeddings.squeeze()

        context_embeddings = self.context_resnet(context_images)
        context_embeddings = context_embeddings.squeeze()

        # we cannot use self.hparams.batch_size here b.c. might be actually smaller e.g. last batch
        batch_size = target_images.shape[0]
        h = torch.zeros(size=(batch_size, self.hparams.lstm_hidden_size), device=self.device)
        c = torch.zeros(size=(batch_size, self.hparams.lstm_hidden_size), device=self.device)

        if not max_length:
            max_length = self.max_length

        # start token
        prev_word = torch.full(size=(batch_size,), fill_value=self.start_token, device=self.device)

        logits = []
        for t in range(max_length):
            word_embedding = self.word_embeddings_dropout(self.word_embeddings(prev_word))
            inputs = torch.cat([target_attributes, target_embeddings, context_embeddings, word_embedding], dim=1)
            h, c = self.lstm_cell(inputs, (h, c))
            logit = self.word_predictor(self.output_dropout(h))
            logits.append(logit)
            if utterances is None:
                prev_word = torch.argmax(logit, dim=1)  # greedy
            else:  # teacher-forcing (assume the ground truth has been generated)
                prev_word = utterances[:, t]  # B x L
        logits = torch.stack(logits, dim=0)
        return logits

    def translate(self, indices: torch.Tensor):
        tokens = []
        for idx in indices:
            if idx == self.end_token:
                break
            if idx == self.start_token or idx == self.pad_token:
                continue
            word = self.vocab.lookup_token(idx)
            if word.startswith("<"):  # remove <>
                word = word[1:-1]
            tokens.append(word)
        return " ".join(tokens)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        logits = self(inputs, utterances=labels, max_length=labels.shape[1])  # align lengths for loss computation
        logits = torch.permute(logits, dims=(1, 2, 0))  # L x B x V -> B x V x L
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_token)
        self.log("loss/data/train", loss, on_step=True, on_epoch=False)
        update_metrics_with_translations(self, self.train_metrics, inputs, labels, logits)
        return loss

    def training_epoch_end(self, step_outputs) -> None:
        # step_or_epoch = self.global_step
        step_or_epoch = self.current_epoch + 1
        log_category_metrics(self, self.train_metrics, "data", "train", step_or_epoch)
        reset_metrics(self.train_metrics)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        logits = self(inputs, max_length=labels.shape[1])
        logits = torch.permute(logits, dims=(1, 2, 0))  # L x B x V -> B x V x L
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_token)
        self.log("loss/data/val", loss, on_step=False, on_epoch=True)
        update_metrics_with_translations(self, self.val_metrics, inputs, labels, logits)

    def validation_epoch_end(self, step_outputs) -> None:
        step_or_epoch = self.global_step  # use step when to be performed more than once an epoch
        # step_or_epoch = self.current_epoch + 1
        logs = log_category_metrics(self, self.val_metrics, "data", "val", step_or_epoch)
        self.log_dict(logs, add_dataloader_idx=False)
        reset_metrics(self.val_metrics)
        # also log possibly intermediate train metrics
        logs = log_category_metrics(self, self.train_metrics, "data", "train", step_or_epoch)
        self.log_dict(logs, add_dataloader_idx=False)
        self.train_metrics["text"].reset()

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

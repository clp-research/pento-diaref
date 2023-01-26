from typing import Any, Dict, Union, Callable
import torch
from torch import nn, Tensor
from torch.nn import functional as F, TransformerEncoderLayer, LayerNorm, TransformerEncoder, TransformerDecoderLayer, \
    TransformerDecoder
import pytorch_lightning as pl
import torchmetrics as tm
from torch.nn.init import xavier_uniform_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from pentodiaref.data.loaders import PentoBoardsDataset, DataMode
from pentodiaref.data.utils import load_vocab, create_vocab, store_vocab
from pentodiaref.metrics import Bleu1Metric, TextLogger, update_metrics_with_translations, \
    log_category_metrics, reset_metrics
import math

from pentodiaref.models.vision.ve import VisualEncoding
from pentodiaref.models.vision.vse import VisualSequenceEncoding


class PositionalEncoding(nn.Module):
    """ The positional encoding for the word utterances """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LitTransformerModel(pl.LightningModule):

    # noinspection PyUnusedLocal
    def __init__(self, model_name: str, data_dir: str, data_mode: DataMode,
                 batch_size: int, d_model: int, n_head: int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int,
                 dropout: float, layer_norm_eps: float, lr: float, l2: float, dry_run: bool = False, **kwargs):
        super(LitTransformerModel, self).__init__()
        self.save_hyperparameters()
        if dry_run:
            print("Dry-run detected!")
        self.dry_run = dry_run
        self.data_dir = data_dir
        self.model_name = data_mode.attach_name_to(model_name)

        self.data_mode = data_mode
        self.max_pieces = 10
        self.max_length = 9 + 1  # 9 + <e> (ignore <s>!)

        if self.data_mode.is_sequential():
            print("Using VisualSequenceEncoding (VSE)")
            self.visual_encoding = VisualSequenceEncoding(d_model, layer_norm_eps, self.device)
        if self.data_mode.is_default():
            print("Using VisualEncoding (VE)")
            self.visual_encoding = VisualEncoding(d_model, layer_norm_eps, self.device)
        self.positional_encoding = PositionalEncoding(d_model, self.max_length, dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, F.relu, layer_norm_eps,
                                    batch_first=False),
            num_encoder_layers, LayerNorm(d_model, eps=layer_norm_eps))

        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, F.relu, layer_norm_eps),
            num_decoder_layers, LayerNorm(d_model, eps=layer_norm_eps))

        # metrics
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

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def configure_callbacks(self):
        monitor_metric = "bleu1/data/val"  # also adjust checkpoint filename!
        filename = self.model_name + "-epoch={epoch:02d}-bleu1={" + monitor_metric + ":.2f}"
        return [
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.EarlyStopping(monitor=monitor_metric, mode="max", patience=20, verbose=True),
            pl.callbacks.ModelCheckpoint(dirpath="saved_models",  # relative to working dir
                                         filename=filename,
                                         monitor=monitor_metric, mode="max", save_top_k=3, verbose=True,
                                         auto_insert_metric_name=False)
        ]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitTransformerModel")
        parser.add_argument("--data_mode", type=DataMode, required=True)
        parser.add_argument("--batch_size", type=int, default=24)
        parser.add_argument("--d_model", type=int, default=512)
        parser.add_argument("--n_head", type=int, default=4)
        parser.add_argument("--num_encoder_layers", type=int, default=3)
        parser.add_argument("--num_decoder_layers", type=int, default=3)
        parser.add_argument("--dim_feedforward", type=int, default=1024)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--layer_norm_eps", type=float, default=1e-5)
        parser.add_argument("--warmup_steps", type=int, default=4000)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--l2", type=float, default=1e-4)
        return parent_parser

    def inputs_to_device(self, d: Dict[str, Tensor]):
        """ This call is necessary, when you manually load the data. """
        for name, x in d.items():
            d[name] = x.to(self.device)
        return d

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
                                            embedding_dim=self.hparams.d_model,
                                            padding_idx=self.vocab(["<p>"])[0])
        self.word_predictor = nn.Linear(in_features=self.hparams.d_model, out_features=(len(self.vocab)))

    def setup(self, stage=None):
        if self.vocab is None:  # for initial training load from hard-drive (store in checkpoint for eval)
            self.vocab = load_vocab(self.data_dir)
        self.pad_token = self.vocab(["<p>"])[0]
        self.start_token = self.vocab(["<s>"])[0]
        self.end_token = self.vocab(["<e>"])[0]

        if stage == "fit" or stage is None:
            if self.dry_run:  # use validation data alone (as it loads faster)
                self.train_data = PentoBoardsDataset(self.vocab, "data", "val", self.data_dir, self.data_mode,
                                                     self.max_pieces)
                self.val_data = PentoBoardsDataset(self.vocab, "data", "val", self.data_dir, self.data_mode,
                                                   self.max_pieces)
                data = self.train_data.get_data()[:10000]
                self.train_data.set_data(data[:8000])
                print("Dry-run data/train:", len(self.train_data))
                self.val_data.set_data(data[8000:])
                print("Dry-run data/val:", len(self.val_data))
            else:
                self.train_data = PentoBoardsDataset(self.vocab, "data", "train", self.data_dir, self.data_mode,
                                                     self.max_pieces)
                self.val_data = PentoBoardsDataset(self.vocab, "data", "val", self.data_dir, self.data_mode,
                                                   self.max_pieces)
        if self.word_embeddings is None:
            self.word_embeddings = nn.Embedding(num_embeddings=len(self.vocab),
                                                embedding_dim=self.hparams.d_model,
                                                padding_idx=self.pad_token)
        if self.word_predictor is None:
            self.word_predictor = nn.Linear(in_features=self.hparams.d_model, out_features=len(self.vocab))

    def configure_optimizers(self):
        initial_lr = self.hparams.d_model ** -0.5
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=initial_lr,
                                      betas=(0.9, 0.98),
                                      eps=1e-8,
                                      weight_decay=self.hparams.l2)
        logger = self.logger.experiment

        def lr_step_fn(step: int) -> float:
            step = 1 if step == 0 else step
            lr_step = step ** -0.5
            lr_warm = step * self.hparams.warmup_steps ** -1.7
            if lr_step < lr_warm:
                lr = lr_step
            else:
                lr = lr_warm
            logger.add_scalar("lr", lr, step)
            return lr

        scheduler = LambdaLR(optimizer, lr_step_fn)
        return [optimizer], [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'reduce_on_plateau': False
            }
        ]

    def predict(self, inputs):
        if self.start_token is None:
            print("Run model.setup('predict')")
            self.setup("predict")
        logits = self(inputs)
        logits = torch.permute(logits, dims=(1, 2, 0))  # L x B x V -> B x V x L
        predictions = torch.argmax(logits, dim=1)  # B x V x L -> B x L
        return [self.translate(prediction) for prediction in predictions]

    def forward(self, inputs, utterances=None, max_length=None) -> Any:
        visual_token_embeddings, visual_padding = self.visual_encoding(inputs)
        #  visual_padding: If a ByteTensor is provided, the non-zero positions are not allowed to attend
        #  while the zero positions will be unchanged.
        visual_padding = (visual_padding == 1)
        context = self.encoder(visual_token_embeddings, src_key_padding_mask=visual_padding)
        # start token
        batch_size = visual_padding.shape[0]
        prev_word = torch.full(size=(batch_size,), fill_value=self.start_token, device=self.device)
        if utterances is not None:  # training
            # first compute the padding mask, because it should be B x L
            # utterances_padding_mask: If a BoolTensor is provided, the positions with the value of True will be ignored
            # while the position with the value of False will be unchanged.
            utterances_padding_mask = (utterances == self.pad_token)  # N x T
            # Then change from B x L -> L x B because batch_first = False
            utterances = torch.permute(utterances, dims=[1, 0])
            # add the start token to the training tgt input
            # shift utterances one to "right"
            # e.g. the label predicts "take the ... <e>" and we input as tgt "<s> take ..."
            prev_word = prev_word.unsqueeze(dim=0)  # B -> 1 x B
            utterances = torch.cat([prev_word, utterances[:-1]], dim=0)
            # utterances_mask: If a FloatTensor is provided, it will be added to the attention weight.
            utterances_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_length).to(self.device)  # T x T
            utterance_embeddings = self.positional_encoding(self.word_embeddings(utterances))
            outputs = self.decoder(utterance_embeddings, context, tgt_mask=utterances_mask,
                                   tgt_key_padding_mask=utterances_padding_mask)
            logits = self.word_predictor(self.output_dropout(outputs))  # map to vocab
        else:  # prediction
            if not max_length:
                max_length = self.max_length
            logits = []
            for t in range(max_length):
                prev_word = prev_word.unsqueeze(dim=0)  # B -> 1 x B
                if t == 0:
                    utterances = prev_word
                else:
                    utterances = torch.cat([utterances, prev_word], dim=0)
                # positional encoding actually expects sequence length first
                utterance_embeddings = self.positional_encoding(self.word_embeddings(utterances))
                current_outputs = self.decoder(utterance_embeddings, context)  # no masking
                time_step_outputs = current_outputs[t]
                time_step_prediction = self.word_predictor(self.output_dropout(time_step_outputs))  # map to vocab
                prev_word = torch.argmax(time_step_prediction, dim=1)  # greedy
                logits.append(time_step_prediction)
            logits = torch.stack(logits, dim=0)
        return logits

    def translate(self, indices: torch.Tensor, keep_special=False):
        tokens = []
        for idx in indices:
            if idx == self.end_token:
                if keep_special:
                    tokens.append("<EOS>")
                break
            if idx == self.start_token or idx == self.pad_token:
                if keep_special:
                    tokens.append("<PAD>")
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

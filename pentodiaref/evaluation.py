from torch.utils.data import DataLoader

from pentodiaref.data.loaders import PentoBoardsDataset
import torch
import pytorch_lightning as pl
import torchmetrics as tm

from pentodiaref.metrics import update_metrics_with_translations, TextMeter


class LitModelEvaluationWrapper(pl.LightningModule):

    def __init__(self, model: pl.LightningModule, stage_name, file_name: str, data_dir: str,
                 results_dir: str, batch_size: int, dry_run: bool = False):
        super(LitModelEvaluationWrapper, self).__init__()
        self.stage_name = stage_name
        self.batch_size = batch_size
        self.model = model
        self.dry_run = dry_run
        self.data_dir = data_dir
        self.data_categories = ["data", "ho-color", "ho-pos", "ho-uts"]
        self.eval_metrics = tm.MetricCollection(TextMeter(stage_name, file_name, results_dir))
        self.eval_data = None

    def configure_optimizers(self):
        return None  # requires but not used

    def setup(self, stage=None):
        assert stage in ["validate", "test"], f"Stage {stage} not in ['validate', 'test']"
        self.model.setup(stage)
        self.model.eval()
        self.model.freeze()
        self.eval_data = [PentoBoardsDataset(self.model.vocab, category, self.stage_name, self.data_dir,
                                             self.model.data_mode, self.model.max_pieces)
                          for category in self.data_categories]

    def validation_step(self, batch, batch_idx, dataset_idx):
        inputs, labels = batch[0], batch[1]
        if len(labels.shape) > 1:  # generation task
            logits = self.model(inputs, max_length=labels.shape[1])  # for cls
            logits = torch.permute(logits, dims=(1, 2, 0))  # L x B x V -> B x V x L
        else:  # classification
            logits = self.model(inputs)
        update_metrics_with_translations(self.model, self.eval_metrics, inputs, labels, logits)

    def validation_epoch_end(self, step_outputs) -> None:
        self.eval_metrics.compute()

    def test_step(self, batch, batch_idx, dataset_idx):
        inputs, labels = batch[0], batch[1]
        if len(labels.shape) > 1:  # generation task
            logits = self.model(inputs, max_length=labels.shape[1])  # for cls
            logits = torch.permute(logits, dims=(1, 2, 0))  # L x B x V -> B x V x L
        else:  # classification
            logits = self.model(inputs)
        update_metrics_with_translations(self.model, self.eval_metrics, inputs, labels, logits)

    def test_epoch_end(self, step_outputs) -> None:
        self.eval_metrics.compute()

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0,
                           collate_fn=dataset.collate) for dataset in self.eval_data]

    def test_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0,
                           collate_fn=dataset.collate) for dataset in self.eval_data]

    def predict_dataloader(self):
        pass

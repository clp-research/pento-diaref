import json
import os
from typing import Optional, Any, Callable, Dict, List

import torchvision
from torch.nn import functional as F

import torchmetrics as tm
import torch
from torchmetrics.text.rouge import ROUGEScore


def log_category_metrics(model, metrics, category, stage_name, step_or_epoch):
    logs = {"step": step_or_epoch}
    for metric_name, metric in metrics.items():
        category_metric = f"{metric_name}/{category}/{stage_name}"
        metric_value = metric.compute()
        if metric_name == "text":
            images_per_row = 2  # default: target and context image
            images = []
            for target, context in metric.get_inputs():
                images.append(target)
                if len(context.shape) > 3:  # a tensor of multiple images
                    for c in context:
                        images.append(c)
                    images_per_row = len(context) + 1
                else:
                    images.append(context)
            if images:
                grid = torchvision.utils.make_grid(images, nrow=images_per_row)
                model.logger.experiment.add_image(f"images/{category}/{stage_name}", grid, step_or_epoch)
            model.logger.experiment.add_text(category_metric, metric_value, step_or_epoch)
        else:
            logs[category_metric] = metric_value
    return logs


def create_category_metrics_with_aggregates(data_categories):
    # We need the aggregates to get the average over all data categories
    # we cannot easily aggregate over "text", so we leave it out
    aggregates = [MeanHoldoutMetricsMetric("rouge1"),
                  MeanHoldoutMetricsMetric("bleu1")]
    metrics = []
    # for each data category there is an individual metric collection
    for category in data_categories:
        metric_collection = tm.MetricCollection({
            "bleu1": Bleu1Metric(),
            "rouge1": Rouge1Metric(),
            "text": TextLogger(store_max=10)
        })
        metrics.append(metric_collection)
        # collect the metrics to aggregate over categories
        for aggregate in aggregates:
            aggregate.update(category, metric_collection)
    return metrics, aggregates


def update_metrics_with_translations(detokenizer, metrics, inputs, labels, logits):
    predictions = torch.argmax(logits, dim=1)  # B x V x L -> B x L
    translations = []
    for prediction, label in zip(predictions, labels):
        translations.append({
            "prediction": detokenizer.translate(prediction),
            "reference": detokenizer.translate(label)
        })
    metrics.update(inputs, logits, labels, translations)


def reset_metrics(metrics):
    for metric_name, metric in metrics.items():
        metric.reset()


class MeanHoldoutMetricsMetric:

    def __init__(self, collect_name, metric_name="ho-total"):
        super().__init__()
        self.collect_name = collect_name
        self.metric_name = metric_name
        self.metrics = []

    def compute(self) -> Any:
        return sum([metric.compute() for metric in self.metrics]) / len(self.metrics)

    def update(self, category: str, metric_collection: tm.MetricCollection) -> None:
        if not category.startswith("ho"):
            return
        for metric_name, metric in metric_collection.items():
            if metric_name == self.collect_name:
                self.metrics.append(metric)


class CrossEntropyMetric(tm.Metric):

    def __init__(self, compute_on_step: bool = False, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.total, self.counts = None, None
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counts", default=torch.tensor(0), dist_reduce_fx="sum")

    def compute(self) -> Any:
        return self.total / self.counts

    def update(self, inputs, logits: torch.Tensor, labels: torch.Tensor, translations: List[Dict]) -> None:
        loss = F.cross_entropy(logits, labels)  # B x V x L <-> B x L
        self.total = self.total + loss.item()
        self.counts += 1


class TextLogger(tm.Metric):

    def __init__(self, store_max=10, compute_on_step: bool = False, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.store_max = store_max
        self.counter = None
        self.table = None
        self.inputs = []
        self.reset()

    def reset(self):
        self.counter = 0
        self.table = ["| ID         | Prediction  | Reference   |"] + \
                     ["| ---------- | ----------- | ----------- |"]
        self.inputs.clear()

    def get_inputs(self):
        return self.inputs

    def compute(self) -> Any:
        table = "\n".join(self.table)
        return table

    def update(self, inputs, logits: torch.Tensor, labels: torch.Tensor, translations: List[Dict]) -> None:
        if self.counter >= self.store_max:
            return
        for targets, contexts, global_id, translation in zip(inputs["target_image"],
                                                             inputs["context_images"],
                                                             inputs["global_id"],
                                                             translations):
            text = translation["prediction"]
            target = translation["reference"]
            self.counter += 1
            if self.counter >= self.store_max:
                return
            self.table.append(f"| {global_id} | {text} | {target} |")
            self.inputs.append((targets, contexts))


class TextMeter(tm.Metric):
    """ Memorize all prediction and store them to a file """

    def __init__(self, stage_name, file_name, target_dir, compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.stage_name = stage_name
        self.file_name = file_name
        self.target_dir = target_dir
        self.predictions = []

    def reset(self):
        self.predictions.clear()

    def compute(self) -> Any:
        file_path = os.path.join(self.target_dir, f"{self.file_name}.{self.stage_name}.predictions.json")
        with open(file_path, "w") as f:
            json.dump(self.predictions, f)
        print("Stored predictions to", file_path)
        return file_path

    def update(self, inputs, logits: torch.Tensor, labels: torch.Tensor, translations: List[Dict]) -> None:
        for global_id, translation in zip(inputs["global_id"], translations):
            self.predictions.append({
                "global_id": global_id.item(),
                "prediction": translation["prediction"],
                "reference": translation["reference"]
            })


class Bleu1Metric(tm.Metric):
    """
        Adapter for BleuScore
    """

    def __init__(self, compute_on_step: bool = False, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.bleu = None
        self.reset()

    def reset(self):
        self.bleu = tm.BLEUScore(n_gram=1, compute_on_step=False)

    def compute(self) -> Any:
        score = self.bleu.compute()
        return score

    def update(self, inputs, logits: torch.Tensor, labels: torch.Tensor, translations: List[Dict]) -> None:
        for translation in translations:
            text = translation["prediction"]
            target = translation["reference"]

            # we remove "easy" start tokens (that are always the same anyway)
            if text.startswith("take the ") and target.startswith("take the "):
                text = text[len("take the "):]
                target = target[len("take the "):]

            self.bleu.update([text], [[target]])


class Rouge1Metric(tm.Metric):
    """
        Adapter for BleuScore
    """

    def __init__(self, compute_on_step: bool = False, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.rouge = None
        self.reset()

    def reset(self):
        self.rouge = ROUGEScore(rouge_keys="rouge1", compute_on_step=False)

    def compute(self) -> Any:
        scores = self.rouge.compute()
        return scores["rouge1_fmeasure"]

    def update(self, inputs, logits: torch.Tensor, labels: torch.Tensor, translations: List[Dict]) -> None:
        for translation in translations:
            text = translation["prediction"]
            target = translation["reference"]

            # we remove "easy" start tokens (that are always the same anyway)
            if text.startswith("take the ") and target.startswith("take the "):
                text = text[len("take the "):]
                target = target[len("take the "):]

            self.rouge.update(text, [target])

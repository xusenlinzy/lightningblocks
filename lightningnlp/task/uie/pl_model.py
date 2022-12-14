from typing import Any, Dict

import torch

from .model import UIE, UIEM
from ...core import TaskTransformer
from ...metrics.extraction import SpanEvaluator
from ...utils.tensor import tensor_to_numpy


class UIEModel(TaskTransformer):
    """Defines ``LightningModule`` for the UIE Task.
    Args:
        *args: :class:`lightningnlp.core.model.TaskTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSequenceClassification``)
        **kwargs: :class:`llightningblocks.core.model.TaskTransformer` arguments.
    """

    def __init__(self, *args, multilingual: bool = False, **kwargs):
        self.multilingual = multilingual
        super(UIEModel, self).__init__(*args, **kwargs)

    def get_auto_model(self, downstream_model_type, downstream_model_name):
        return UIEM if self.multilingual else UIE

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def common_step(self, batch: Any):
        outputs = self.model(**batch)
        start_prob = tensor_to_numpy(outputs.start_prob)
        end_prob = tensor_to_numpy(outputs.end_prob)

        start_ids = tensor_to_numpy(batch['start_positions'].to(torch.float32))
        end_ids = tensor_to_numpy(batch['end_positions'].to(torch.float32))

        num_correct, num_infer, num_label = self.metrics.compute(
            start_prob, end_prob, start_ids, end_ids,
        )
        self.metrics.update(num_correct, num_infer, num_label)

    def common_epoch_end(self, prefix: str):
        metric_dict = self.compute_metrics(mode=prefix)
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        return metric_dict

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return self.common_step(batch)

    def validation_epoch_end(self, outputs):
        return self.common_epoch_end("val")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return self.common_step(batch)

    def test_epoch_end(self, outputs):
        return self.common_epoch_end("val")

    def configure_metrics(self, _) -> None:
        self.metric = SpanEvaluator()

    def compute_metrics(self, mode="val") -> Dict[str, float]:
        p, r, f = self.metric.accumulate()
        self.metrics.reset()
        return {f"{mode}_precision": p, f"{mode}_recall": r, f"{mode}_f1": f}

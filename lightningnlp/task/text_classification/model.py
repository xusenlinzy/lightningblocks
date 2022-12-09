from typing import Any, Dict, Optional

import torch
from torchmetrics import Accuracy, Precision, Recall

from lightningnlp.core import TaskTransformer
from lightningnlp.task.text_classification.auto import get_auto_tc_model_config, get_auto_tc_model


class TextClassificationTransformer(TaskTransformer):
    """Defines ``LightningModule`` for the Text Classification Task.
    Args:
        *args: :class:`lightningnlp.core.model.TaskTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSequenceClassification``)
        **kwargs: :class:`llightningblocks.core.model.TaskTransformer` arguments.
    """

    def __init__(
        self,
        downstream_model_type: str,
        downstream_model_name: str,
        label_map: Dict[Any, int],
        model_config_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        model_config_kwargs = model_config_kwargs or {}
        model_config_kwargs = get_auto_tc_model_config(label_map, model_config_kwargs)
        super().__init__(downstream_model_type, downstream_model_name,
                         model_config_kwargs=model_config_kwargs, **kwargs)
        self.metrics = {}

    def get_auto_model(self, downstream_model_type, downstream_model_name):
        return get_auto_tc_model(model_name=downstream_model_name, model_type=downstream_model_type)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        if batch["labels"] is not None:
            metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
            self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if -1 in batch["labels"]:
            batch["labels"] = None
        return self.common_step("test", batch)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        batch["labels"] = None
        outputs = self.model(**batch)
        logits = outputs.logits
        return torch.argmax(logits, dim=1)

    def configure_metrics(self, _) -> None:
        self.prec = Precision(num_classes=self.num_classes, average="macro")
        self.recall = Recall(num_classes=self.num_classes, average="macro")
        self.acc = Accuracy()
        self.metrics = {
            "precision": self.prec,
            "recall": self.recall,
            "accuracy": self.acc,
        }

    def compute_metrics(self, preds, labels, mode="val") -> Dict[str, torch.Tensor]:
        # Not required by all models. Only required for classification
        return {f"{mode}_{k}": metric(preds, labels) for k, metric in self.metrics.items()}

    @property
    def pipeline_task(self) -> str:
        return "sentiment-analysis"

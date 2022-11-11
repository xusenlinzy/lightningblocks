from functools import partial
from typing import Any

import torch

from lightningnlp.core import TaskTransformer
from lightningnlp.metrics import SquadMetric
from lightningnlp.task.question_answering.data import QuestionAnsweringDataModule


# noinspection PyUnresolvedReferences
class QuestionAnsweringTransformer(TaskTransformer):
    """Defines ``LightningModule`` for the Question Answering Task.
    """

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    @property
    def pipeline_task(self) -> str:
        return "question-answering"

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        batch.pop("offset_mapping")
        example_ids = batch.pop("example_id")
        outputs = self.model(**batch)
        self.metric.update(example_ids, outputs.start_logits, outputs.end_logits)

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()

    def on_validation_epoch_end(self) -> None:
        metric_dict = self.metric.compute()
        self.log_dict(metric_dict, prog_bar=True)

    def configure_metrics(self, stage: str):
        dataset: QuestionAnsweringDataModule = self.trainer.datamodule
        validation_dataset = dataset.ds["validation"]
        original_validation_dataset = dataset.ds["validation_original"]
        postprocess_func = partial(
            dataset.postprocess_func,
            dataset=dataset.ds,
            validation_dataset=validation_dataset,
            original_validation_dataset=original_validation_dataset,
        )
        example_id_strings = dataset.example_id_strings
        self.metric = SquadMetric(postprocess_func=postprocess_func, example_id_strings=example_id_strings)

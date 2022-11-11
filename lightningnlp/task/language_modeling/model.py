from typing import Any

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers import AutoModelForCausalLM

from lightningnlp.core import TaskTransformer


class LanguageModelingTransformer(TaskTransformer):
    """Defines ``LightningModule`` for the Language Modeling Task.
    """

    def get_auto_model(self, downstream_model_type, downstream_model_name):
        return AutoModelForCausalLM

    def on_fit_start(self):
        tokenizer_length = len(self.tokenizer)
        self.model.resize_token_embeddings(tokenizer_length)

    def _step(self, batch):
        outputs = self.model(**batch)
        return outputs[0]

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, sync_dist=True)

    @property
    def pipeline_task(self) -> str:
        return "text-generation"

    def generate(self, text: str, device: torch.device = torch.device("cpu"), **kwargs) -> Any:
        if self.tokenizer is None:
            raise MisconfigurationException(
                "A tokenizer is required to use the `generate` function. "
                "Please pass a tokenizer `LanguageModelingTransformer(tokenizer=...)`."
            )
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = inputs.to(device)
        return self.model.generate(inputs["input_ids"], **kwargs)

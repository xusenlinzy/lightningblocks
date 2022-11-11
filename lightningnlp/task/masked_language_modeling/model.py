from transformers import AutoModelForMaskedLM

from lightningnlp.core import TaskTransformer


class MaskedLanguageModelingTransformer(TaskTransformer):
    """Defines ``LightningModule`` for the Masked Language Modeling Task.
    """

    def get_auto_model(self, downstream_model_type, downstream_model_name):
        return AutoModelForMaskedLM

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
        return "fill-mask"

import pytorch_lightning as pl

from lightningnlp.core import TransformersBaseFinetuning


class FreezeEmbeddings(TransformersBaseFinetuning):
    """Freezes the embedding layers during training."""

    def __init__(self, train_bn: bool = True):
        super().__init__("", train_bn)

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        is_t5 = pl_module.model.config.model_type in ["t5", "mt5"]
        model = pl_module.model if is_t5 else pl_module.model.model
        self.freeze(modules=model.shared, train_bn=self.train_bn)
        for layer in (model.encoder, model.decoder):
            self.freeze(layer.embed_tokens)
            if not is_t5:
                self.freeze(layer.embed_positions)

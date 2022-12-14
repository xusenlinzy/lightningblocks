from torchmetrics.text.bleu import BLEUScore
from transformers import MBartTokenizer, AutoModelForSeq2SeqLM

from .data import TranslationDataModule
from ...core.seq2seq.model import Seq2SeqTransformer


# noinspection PyUnresolvedReferences
class TranslationTransformer(Seq2SeqTransformer):
    """Defines ``LightningModule`` for the Translation Task.
    Args:
        *args: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSeq2SeqLM``)
        n_gram: Gram value ranged from 1 to 4.
        smooth: Whether or not to apply smoothing.
        **kwargs: :class:`lightning_transformers.core.model.TaskTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        n_gram: int = 4,
        smooth: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bleu = None
        self.n_gram = n_gram
        self.smooth = smooth

    def get_auto_model(self, downstream_model_type, downstream_model_name):
        return AutoModelForSeq2SeqLM

    def compute_generate_metrics(self, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        # wrap targets in list as score expects a list of potential references
        result = self.bleu(preds=pred_lns, target=tgt_lns)
        self.log(f"{prefix}_bleu_score", result, on_step=False, on_epoch=True, prog_bar=True)

    def configure_metrics(self, stage: str):
        self.bleu = BLEUScore(self.n_gram, self.smooth)

    def initialize_model(self, pretrained_model_name_or_path, config, model):
        super().initialize_model(pretrained_model_name_or_path, config, model)
        if isinstance(self.tokenizer, MBartTokenizer) and self.model.config.decoder_start_token_id is None:
            dm: TranslationDataModule = self.trainer.datamodule
            tgt_lang = dm.target_language
            assert tgt_lang is not None, "mBart requires --target_language"
            self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[tgt_lang]

    @property
    def hf_pipeline_task(self) -> str:
        return "translation_xx_to_yy"

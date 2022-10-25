from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoModelForSeq2SeqLM
from lightningnlp.core.seq2seq.model import Seq2SeqTransformer


class SummarizationTransformer(Seq2SeqTransformer):
    """Defines ``LightningModule`` for the Summarization Task.
    Args:
        *args: :class:`lightningnlp.core.model.TaskTransformer` arguments.
        use_stemmer: Use Porter stemmer to strip word suffixes to improve matching.
        **kwargs: :class:`lightningnlp.core.model.TaskTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        use_stemmer: bool = True,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rouge = None
        self.use_stemmer = use_stemmer

    def get_auto_model(self, downstream_model_type, downstream_model_name):
        return AutoModelForSeq2SeqLM

    def compute_generate_metrics(self, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        result = self.rouge(pred_lns, tgt_lns)
        self.log_dict(result, on_step=False, on_epoch=True)

    def configure_metrics(self, stage: str):
        self.rouge = ROUGEScore(use_stemmer=self.use_stemmer)

    @property
    def pipeline_task(self) -> str:
        return "summarization"

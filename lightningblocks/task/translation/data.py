from typing import Tuple
from lightningblocks.core.seq2seq.data import Seq2SeqDataModule


class TranslationDataModule(Seq2SeqDataModule):
    """Defines the ``LightningDataModule`` for Translation Datasets.
    Args:
        *args: ``Seq2SeqDataModule`` specific arguments.
        **kwargs: ``Seq2SeqDataModule`` specific arguments.
    """

    def __init__(self, *args, source_language: str = "", target_language: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.source_language = source_language
        self.target_language = target_language

    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return self.source_language, self.target_language

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional

import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from lightningblocks.core import TransformerDataModule


# noinspection PyUnresolvedReferences
class QuestionAnsweringDataModule(TransformerDataModule):
    """Defines the ``LightningDataModule`` for Question Answering Datasets.
    Args:
        *args: ``HFDataModule`` specific arguments.
        max_length: The maximum total input sequence length after tokenization. Sequences longer
            than this will be truncated, sequences shorter will be padded.
        version_2_with_negative: If true, some of the examples do not have an answer.
        null_score_diff_threshold: The threshold used to select the null answer:
            if the best answer has a score that is less than
            the score of the null answer minus this threshold, the null answer is selected for this example.
            Only useful when `version_2_with_negative=True`.
        doc_stride: When splitting up a long document into chunks, how much stride to take between chunks.
        n_best_size: The total number of n-best predictions to generate when looking for an answer.
        max_answer_length: The maximum length of an answer that can be generated. This is needed because the start
            and end predictions are not conditioned on one another.
        output_dir: If provided, the dictionaries of predictions,
            n_best predictions (with their scores and logits) and,
            if `version_2_with_negative=True`, the dictionary of the scores differences
            between best and null answers, are saved in `output_dir`.
        **kwargs: ``HFDataModule`` specific arguments.
    """

    def __init__(
        self,
        *args,
        max_length: int = 384,
        version_2_with_negative: bool = False,
        null_score_diff_threshold: float = 0.0,
        doc_stride: int = 128,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(*args, max_length=max_length, **kwargs)
        self.example_id_strings = OrderedDict()
        self.version_2_with_negative = version_2_with_negative
        self.null_score_diff_threshold = null_score_diff_threshold
        self.doc_stride = doc_stride
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.output_dir = output_dir

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        train = stage == "fit"
        column_names = dataset["train" if train else "validation"].column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]
        self.answer_column_name = answer_column_name

        kwargs = {
            "tokenizer": self.tokenizer,
            "pad_on_right": self.tokenizer.padding_side == "right",
            "question_column_name": question_column_name,
            "context_column_name": context_column_name,
            "answer_column_name": answer_column_name,
            "max_length": self.max_length,
            "doc_stride": self.doc_stride,
            "padding": False,
        }

        prepare_train_features = partial(self.convert_to_train_features, **kwargs)

        if train:
            dataset["train"] = dataset["train"].map(
                prepare_train_features,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=self.load_from_cache_file,
            )

        if "test" not in dataset:
            kwargs.pop("answer_column_name")
            prepare_validation_features = partial(
                self.convert_to_validation_features, example_id_strings=self.example_id_strings, **kwargs
            )
            dataset["validation_original"] = dataset["validation"]  # keep an original copy for computing metrics
            dataset["validation"] = dataset["validation"].map(
                prepare_validation_features,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                # Legacy code `prepare_validation_features` must run to populate `self.example_id_strings`
                # Therefore we cannot load from cache here
                load_from_cache_file=False,
            )
        return dataset

    @property
    def collate_fn(self) -> Callable:
        return DataCollatorWithPadding(self.tokenizer)

    @staticmethod
    def convert_to_train_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        pad_on_right: bool,
        question_column_name: str,
        context_column_name: str,
        answer_column_name: str,
        max_length: int,
        doc_stride: int,
        padding: bool,
    ):
        raise NotImplementedError

    @staticmethod
    def convert_to_validation_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        pad_on_right: bool,
        question_column_name: str,
        context_column_name: str,
        max_length: int,
        doc_stride: int,
        padding: bool,
    ):
        raise NotImplementedError

    def postprocess_func(
        self,
        dataset: Dataset,
        validation_dataset: Dataset,
        original_validation_dataset: Dataset,
        predictions: Dict[int, torch.Tensor],
    ) -> Any:
        raise NotImplementedError

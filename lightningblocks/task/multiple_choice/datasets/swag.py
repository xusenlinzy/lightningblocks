from functools import partial
from typing import Any, Dict, Optional
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from lightningblocks.task.multiple_choice.data import MultipleChoiceDataModule


# noinspection PyUnresolvedReferences
class SwagMultipleChoiceDataModule(MultipleChoiceDataModule):
    num_choices: int = 4  # there are four different endings to select in the SWAG datasets

    def __init__(self, *args, dataset_name: str = "swag", dataset_config_name="regular", **kwargs):
        super().__init__(*args, dataset_name=dataset_name, dataset_config_name=dataset_config_name, **kwargs)

    @property
    def num_classes(self) -> int:
        return len(self.ending_column_names)

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            num_choices=self.num_choices,
            context_name=self.context_name,
            question_header_name=self.question_header_name,
            ending_names=self.ending_column_names,
            max_length=self.max_length,
        )

        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )

        cols_to_keep = [
            x
            for x in ["input_ids", "attention_mask", "token_type_ids", "label", "idx"]
            if x in dataset["train"].features
        ]
        dataset.set_format(columns=cols_to_keep)

        return dataset

    @property
    def ending_column_names(self) -> list:
        return [f"ending{i}" for i in range(self.num_choices)]

    @property
    def context_name(self) -> str:
        return "sent1"

    @property
    def question_header_name(self) -> str:
        return "sent2"

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        num_choices: int,
        context_name: str,
        question_header_name: str,
        ending_names: list,
        max_length: int,
    ) -> Dict:
        first_sentences = [[context] * num_choices for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences, second_sentences, truncation=True, max_length=max_length, padding=False,
        )
        # Un-flatten
        return {
            k: [v[i: i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()
        }

    def test_dataloader(self) -> Optional[DataLoader]:
        """SWAG does not offer labels within the test set (blind)."""
        pass

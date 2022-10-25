from functools import partial
from typing import Any, Optional, List

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from pytorch_lightning.utilities import rank_zero_warn

from lightningblocks.core import TransformerDataModule


class TokenClassificationDataModule(TransformerDataModule):
    """Defines the ``LightningDataModule`` for Token Classification Datasets.
    Args:
        *args: ``HFDataModule`` specific arguments.
        **kwargs: ``HFDataModule`` specific arguments.
    """

    def __init__(self, *args, task_name: str = "crf", is_chinese: bool = True, **kwargs) -> None:
        self.labels = None
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.is_chinese = is_chinese

    def get_process_fct(self, text_column_name, label_column_name, mode):
        convert_to_features = partial(
            TokenClassificationDataModule.convert_to_features,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            label_to_id=self.label_to_id,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            mode=mode,
            is_chinese=self.is_chinese,
        )
        return convert_to_features

    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        label_column_name, text_column_name = self._setup_input_fields(dataset, stage)
        self._prepare_labels(dataset, label_column_name)

        convert_to_features_train = self.get_process_fct(text_column_name, label_column_name, "train")
        convert_to_features_val = self.get_process_fct(text_column_name, label_column_name, "val")

        train_dataset = dataset["train"].map(
            convert_to_features_train,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Running tokenizer on train datasets",
            new_fingerprint=f"train-{self.max_length}-{self.task_name}",
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )

        def process_dev(example):
            return {
                "target": {
                    (ent['label'], str(ent['start_offset']), str(ent['end_offset']), ent['entity'])
                    for ent in example[label_column_name]
                }
            }

        val_dataset = dataset["validation"].map(process_dev)
        val_dataset = val_dataset.map(
            convert_to_features_val,
            batched=True,
            remove_columns=[label_column_name],
            desc="Running tokenizer on validation datasets",
            new_fingerprint=f"validation-{self.max_length}-{self.task_name}",
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )

        if "test" in dataset:
            test_dataset = dataset["test"].map(process_dev)
            test_dataset = test_dataset.map(
                convert_to_features_val,
                batched=True,
                remove_columns=[label_column_name],
                desc="Running tokenizer on test datasets",
                new_fingerprint=f"test-{self.max_length}-{self.task_name}",
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=self.load_from_cache_file,
            )

        all_dataset = {"train": train_dataset, "validation": val_dataset}

        return all_dataset.update({"test": test_dataset}) if "test" in dataset else all_dataset

    def _setup_input_fields(self, dataset, stage):
        split = "train" if stage == "fit" else "validation"
        column_names = dataset[split].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        label_column_name = "entities" if "entities" in column_names else column_names[1]
        return label_column_name, text_column_name

    # noinspection PyPropertyAccess
    def _prepare_labels(self, dataset, label_column_name):
        # Create unique label set from train datasets.
        label_list = sorted({label["label"] for column in dataset["train"][label_column_name] for label in column})
        label_to_id = {l: i for i, l in enumerate(label_list)}
        self.labels = label_list
        self.label_to_id = label_to_id

    @property
    def label_list(self) -> List[str]:
        if self.labels is None:
            rank_zero_warn("Labels has not been set, calling `setup('fit')`.")
            self.setup("fit")
        return self.labels

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        label_to_id,
        text_column_name,
        label_column_name,
        mode,
        is_chinese,
    ):

        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]

        tokenized_inputs = tokenizer(
            sentences,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_token_type_ids=False,
            return_offsets_mapping=True,
        )

        if mode == "train":
            labels = []
            for i, entity_list in enumerate(examples[label_column_name]):
                res = []
                for _ent in entity_list:
                    try:
                        start = tokenized_inputs.char_to_token(i, _ent['start_offset'])
                        end = tokenized_inputs.char_to_token(i, _ent['end_offset'] - 1)
                    except Exception:
                        continue
                    if start is None or end is None:
                        continue
                    res.append([start, end, label_to_id[_ent['label']]])
                labels.append(res)
            tokenized_inputs["labels"] = labels

        return tokenized_inputs

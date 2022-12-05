from functools import partial
from typing import Any, Optional

from datasets import Dataset
from pytorch_lightning.utilities import rank_zero_warn
from transformers import PreTrainedTokenizerBase

from lightningnlp.core import TransformerDataModule


class RelationExtractionDataModule(TransformerDataModule):
    """Defines the ``LightningDataModule`` for Relation Extraction Datasets.
    Args:
        *args: ``HFDataModule`` specific arguments.
        **kwargs: ``HFDataModule`` specific arguments.
    """

    def __init__(self, *args, task_name: str = "casrel", is_chinese: bool = True, **kwargs) -> None:
        self.predicates = None
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.is_chinese = is_chinese

    def get_process_fct(self, text_column_name, label_column_name, mode):
        convert_to_features = partial(
            RelationExtractionDataModule.convert_to_features,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            mode=mode,
            is_chinese=self.is_chinese,
        )
        return convert_to_features

    # noinspection PyUnresolvedReferences
    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        label_column_name, text_column_name = self._setup_input_fields(dataset, stage)
        self._prepare_labels(dataset, label_column_name)

        convert_to_features_train = self.get_process_fct(text_column_name, label_column_name, "train")
        convert_to_features_val = self.get_process_fct(text_column_name, label_column_name, "val")

        train_dataset = self.process_train(dataset["train"], predicate2id=self.predicate_to_id)
        train_dataset = train_dataset.map(
            convert_to_features_train,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Running tokenizer on train datasets",
            new_fingerprint=f"train-{self.max_length}-{self.task_name}",
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )

        def process_dev(example):
            triplet = [[spo["predicate"], spo["subject"], spo["object"]] for spo in example["spo_list"]]
            return {"text": example["text"], "target": triplet}

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

        all_dataset = {"train": train_dataset, "validation": val_dataset}

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

            all_dataset.update({"test": test_dataset})

        return all_dataset

    def _setup_input_fields(self, dataset, stage):
        split = "train" if stage == "fit" else "validation"
        column_names = dataset[split].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        label_column_name = "spo_list" if "spo_list" in column_names else column_names[1]
        return label_column_name, text_column_name

    # noinspection PyPropertyAccess
    def _prepare_labels(self, dataset, label_column_name):
        # Create unique label set from train datasets.
        label_list = sorted({label["predicate"] for column in dataset["train"][label_column_name] for label in column})
        label_to_id = {l: i for i, l in enumerate(label_list)}
        self.predicates = label_list
        self.predicate_to_id = label_to_id

    @property
    def predicate_list(self) -> int:
        if self.predicates is None:
            rank_zero_warn("Labels has not been set, calling `setup('fit')`.")
            self.setup("fit")
        return self.predicates

    @staticmethod
    def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        return next((i for i in range(len(sequence)) if sequence[i: i + n] == pattern), -1)

    def process_train(self, ds, predicate2id):
        def judge(example):
            spo_list = []
            for spo in example["spo_list"]:
                sub = self.search(spo["subject"], example["text"])
                obj = self.search(spo["object"], example["text"])
                if sub == -1 or obj == -1:
                    continue
                else:
                    spo_list.append([1])
            return len(spo_list) > 0

        def convert(example):
            spo_list = []
            for spo in example["spo_list"]:
                sub = self.search(spo["subject"], example["text"])
                pre = predicate2id[spo["predicate"]]
                obj = self.search(spo["object"], example["text"])
                if sub == -1 or obj == -1:
                    continue
                else:
                    spo_list.append(
                        [
                            sub,
                            sub + len(spo["subject"]) - 1,
                            pre,
                            obj,
                            obj + len(spo["object"]) - 1,
                        ]
                    )

            assert spo_list
            return {"text": example["text"], "spo_list": spo_list}

        return ds.filter(judge).map(convert)

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
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
            for i, spo_list in enumerate(examples[label_column_name]):
                spo = []
                for _sh, _st, p, _oh, _ot in spo_list:
                    try:
                        sh = tokenized_inputs.char_to_token(i, _sh)
                        oh = tokenized_inputs.char_to_token(i, _oh)
                        st = tokenized_inputs.char_to_token(i, _st)
                        ot = tokenized_inputs.char_to_token(i, _ot)
                    except Exception:
                        print("char_to_token error!")
                        continue
                    if sh is None or oh is None or st is None or ot is None:
                        continue
                    spo.append([sh, st, p, oh, ot])
                labels.append(spo)
            tokenized_inputs["labels"] = labels

        return tokenized_inputs

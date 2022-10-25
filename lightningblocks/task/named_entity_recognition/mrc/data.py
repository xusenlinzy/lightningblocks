import torch
from functools import partial
from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from lightningblocks.task.named_entity_recognition.data import TokenClassificationDataModule


@dataclass
class DataCollatorForMRCNer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None
    ignore_list: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        flattened_features = [
            [{k: v[i] for k, v in feature.items() if k not in self.ignore_list} for i in range(self.num_labels)] for feature
            in features]
        flattened_features = sum(flattened_features, [])
        labels = [feature.pop("labels") for feature in flattened_features] if "labels" in flattened_features[
            0].keys() else None

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature["text"] for feature in features for _ in range(self.num_labels)]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature["offset_mapping"][i] for feature in features for i in
                                           range(self.num_labels)]
            if "target" in features[0].keys():
                batch['target'] = [{tuple([t[0], int(t[1]), int(t[2]), t[3]]) for t in feature.pop("target")} for
                                   feature in features]
            return batch

        batch_start_positions = torch.zeros_like(batch["input_ids"])
        batch_end_positions = torch.zeros_like(batch["input_ids"])
        for i, lb in enumerate(labels):
            for start, end in lb:
                batch_start_positions[i, start] = 1
                batch_end_positions[i, end] = 1

        batch['start_positions'] = batch_start_positions
        batch['end_positions'] = batch_end_positions
        return batch


class MRCNerDataModule(TokenClassificationDataModule):
    """MRC model for Named Entity Recognition.
    """
    def __init__(self, schema2prompt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.schema2prompt = schema2prompt

    def get_process_fct(self, text_column_name, label_column_name, mode):
        convert_to_features = partial(
            MRCNerDataModule.convert_to_features,
            schema2prompt=self.schema2prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            mode=mode,
            is_chinese=self.is_chinese,
        )
        return convert_to_features

    @staticmethod
    def convert_to_features(
        examples: Any,
        schema2prompt: Dict,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        text_column_name,
        label_column_name,
        mode,
        is_chinese,
    ):

        first_sentences = [list(schema2prompt.values()) for _ in examples[text_column_name]]
        second_sentences = [[t] * len(schema2prompt) for t in examples[text_column_name]]

        # flatten everthing
        first_sentences, second_sentences = sum(first_sentences, []), sum(second_sentences, [])
        if is_chinese:
            second_sentences = [text.replace(" ", "-") for text in second_sentences]

        tokenized_inputs = tokenizer(
            first_sentences,
            second_sentences,
            max_length=max_length,
            padding=False,
            truncation='only_second',
            return_offsets_mapping=True,
        )

        if mode == "train":
            all_label_dict = []
            for entity_list in examples[label_column_name]:
                label_dict = {k: [] for k in schema2prompt.keys()}
                for _ent in entity_list:
                    label_dict[_ent['label']].append((_ent['start_offset'], _ent['end_offset']))
                all_label_dict.extend(list(label_dict.values()))

            labels = []
            for i, lb in enumerate(all_label_dict):
                res = []
                input_ids = tokenized_inputs["input_ids"][i]
                offset_mapping = tokenized_inputs["offset_mapping"][i]
                # 区分prompt和text
                sequence_ids = tokenized_inputs.sequence_ids(i)
                for start, end in lb:
                    # 找到token级别的index start
                    token_start_index = 0
                    while sequence_ids[token_start_index] != 1:
                        token_start_index += 1
                    # 找到token级别的index end
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != 1:
                        token_end_index -= 1
                    # 检测答案是否在文本区间的外部
                    if (offset_mapping[token_start_index][0] <= start) and (
                            offset_mapping[token_end_index][1] >= end):
                        while token_start_index < len(offset_mapping) and offset_mapping[token_start_index][0] <= start:
                            token_start_index += 1
                        while offset_mapping[token_end_index][1] >= end and token_end_index > 0:
                            token_end_index -= 1
                        res.append((token_start_index - 1, token_end_index + 1))
                labels.append(res)
            tokenized_inputs["labels"] = labels

        return {k: [v[i: i + len(schema2prompt)] for i in range(0, len(v), len(schema2prompt))] for k, v in
                tokenized_inputs.items()}

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForMRCNer(tokenizer=self.tokenizer, num_labels=len(self.labels), ignore_list=ignore_list)

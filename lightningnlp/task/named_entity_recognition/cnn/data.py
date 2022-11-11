from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, List, Any, Dict

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from lightningnlp.task.named_entity_recognition.data import TokenClassificationDataModule
from lightningnlp.utils.tensor import sequence_padding


@dataclass
class DataCollatorForCNNNer:
    num_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        matrix = ([feature.pop("label") for feature in features] if "label" in features[0].keys() else None)

        input_ids = [feature.pop("input_ids") for feature in features]
        input_ids = torch.from_numpy(sequence_padding(input_ids))

        indexes = [feature.pop("indexes") for feature in features]
        indexes = torch.from_numpy(sequence_padding(indexes))

        batch = {"input_ids": input_ids, "indexes": indexes}

        if matrix is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple([t[0], int(t[1]), int(t[2]), t[3]]) for t in feature.pop("target")} for
                                   feature in features]
            return batch

        seq_len = max(len(i) for i in matrix)
        matrix_new = np.ones((input_ids.shape[0], seq_len, seq_len, self.num_labels)) * -100
        for i in range(input_ids.shape[0]):
            matrix_new[i, :len(matrix[i][0]), :len(matrix[i][0]), :] = matrix[i]
        matrix = torch.from_numpy(matrix_new).long()

        batch["labels"] = matrix

        return batch


class CNNNerDataModule(TokenClassificationDataModule):
    """CNN for Named Entity Recognition.
    """

    def get_process_fct(self, text_column_name, label_column_name, mode):
        convert_to_features = partial(
            CNNNerDataModule.convert_to_features,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            label_to_id=self.label_to_id,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            is_chinese=self.is_chinese,
            mode=mode,
        )
        return convert_to_features

    @staticmethod
    def convert_to_features(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        label_to_id,
        text_column_name,
        label_column_name,
        is_chinese,
        mode,
    ):

        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]

        input_keys = ["input_ids", "indexes", "label"] if mode == "train" else ["input_ids", "indexes"]
        encoded_inputs = {k: [] for k in input_keys}

        def get_new_ins(bpes, spans, indexes):
            bpes.append(tokenizer.sep_token_id)
            cur_word_idx = indexes[-1]
            indexes.append(0)

            if spans is not None:
                matrix = np.zeros((cur_word_idx, cur_word_idx, len(label_to_id)), dtype=np.int8)
                for _ner in spans:
                    s, e, t = _ner
                    if s <= e < cur_word_idx:
                        matrix[s, e, t] = 1
                        matrix[e, s, t] = 1
                return bpes, indexes, matrix

            return bpes, indexes

        for i in range(len(sentences)):
            sentence = sentences[i]
            spans = [] if mode == "train" else None
            _indexes = []
            _bpes = []

            for idx, word in enumerate(sentence):
                __bpes = tokenizer.encode(word, add_special_tokens=False)
                _indexes.extend([idx] * len(__bpes))
                _bpes.extend(__bpes)

            indexes = [0] + [i + 1 for i in _indexes]
            bpes = [tokenizer.cls_token_id] + _bpes

            if len(bpes) > max_length - 1:
                indexes = indexes[:max_length - 1]
                bpes = bpes[:max_length - 1]

            if mode == "train":
                label = examples[label_column_name][i]
                spans = [(ent["start_offset"], ent["end_offset"] - 1, label_to_id.get(ent["label"]),) for ent in label]

            for k, v in zip(input_keys, get_new_ins(bpes, spans, indexes)):
                encoded_inputs[k].append(v)

        return encoded_inputs

    @property
    def collate_fn(self) -> Optional[Callable]:
        return DataCollatorForCNNNer(num_labels=len(self.labels))

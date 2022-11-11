import itertools
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, List, Any, Dict

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from lightningnlp.task.named_entity_recognition.data import TokenClassificationDataModule
from lightningnlp.utils.tensor import sequence_padding

# dist_inputs
# https://github.com/ljynlp/W2NER/issues/17
DIST_TO_IDX = torch.zeros(1000, dtype=torch.int64)
DIST_TO_IDX[1] = 1
DIST_TO_IDX[2:] = 2
DIST_TO_IDX[4:] = 3
DIST_TO_IDX[8:] = 4
DIST_TO_IDX[16:] = 5
DIST_TO_IDX[32:] = 6
DIST_TO_IDX[64:] = 7
DIST_TO_IDX[128:] = 8
DIST_TO_IDX[256:] = 9


@dataclass
class DataCollatorForW2Ner:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("grid_label") for feature in features] if "grid_label" in features[0].keys() else None)

        input_ids = [feature.pop("input_ids") for feature in features]
        input_ids = torch.from_numpy(sequence_padding(input_ids))

        pieces2word = [feature.pop("pieces2word") for feature in features]
        input_lengths = torch.tensor([len(i) for i in pieces2word], dtype=torch.long)

        max_wordlen = torch.max(input_lengths).item()
        max_pieces_len = max([x.shape[0] for x in input_ids])

        batch_size = input_ids.shape[0]
        sub_mat = torch.zeros(batch_size, max_wordlen, max_pieces_len, dtype=torch.long)
        pieces2word = self.fill(pieces2word, sub_mat)

        dist_mat = torch.zeros(batch_size, max_wordlen, max_wordlen, dtype=torch.long)
        dist_inputs = [feature.pop("dist_inputs") for feature in features]
        dist_inputs = self.fill(dist_inputs, dist_mat)

        mask_mat = torch.zeros(batch_size, max_wordlen, max_wordlen, dtype=torch.long)
        grid_mask = [feature.pop("grid_mask") for feature in features]
        grid_mask = self.fill(grid_mask, mask_mat)

        batch = {
            "input_ids": input_ids,
            "dist_inputs": dist_inputs,
            "pieces2word": pieces2word,
            "grid_mask": grid_mask,
            "input_lengths": input_lengths,
        }

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple([t[0], int(t[1]), int(t[2]), t[3]]) for t in feature.pop("target")} for
                                   feature in features]
            return batch

        labels_mat = torch.zeros(batch_size, max_wordlen, max_wordlen, dtype=torch.long)
        labels = self.fill(labels, labels_mat)
        batch["grid_labels"] = labels

        return batch

    @staticmethod
    def fill(data, new_data):
        for i, d in enumerate(data):
            if isinstance(d, np.ndarray):
                new_data[i, :len(d), :len(d[0])] = torch.from_numpy(d).long()
            else:
                new_data[i, :len(d), :len(d[0])] = torch.tensor(d, dtype=torch.long)
        return new_data


# noinspection PyUnboundLocalVariable,PyAssignmentToLoopOrWithParameter
class W2NerDataModule(TokenClassificationDataModule):
    """W2Ner model for Named Entity Recognition.
    """

    def get_process_fct(self, text_column_name, label_column_name, mode):
        if mode == "train":
            convert_to_features = partial(
                W2NerDataModule.convert_to_features_train,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                label_to_id=self.label_to_id,
                text_column_name=text_column_name,
                label_column_name=label_column_name,
                is_chinese=self.is_chinese,
            )
        else:
            convert_to_features = partial(
                W2NerDataModule.convert_to_features_valid,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                text_column_name=text_column_name,
                is_chinese=self.is_chinese,
            )
        return convert_to_features

    @staticmethod
    def convert_to_features_train(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        label_to_id,
        text_column_name,
        label_column_name,
        is_chinese,
    ):

        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]

        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask", "grid_label"]
        encoded_inputs = {k: [] for k in input_keys}

        for sentence, label in zip(sentences, examples[label_column_name]):
            tokens = [tokenizer.tokenize(word) for word in sentence[:max_length - 2]]
            pieces = [piece for pieces in tokens for piece in pieces]
            _input_ids = tokenizer.convert_tokens_to_ids(pieces)
            _input_ids = np.array([tokenizer.cls_token_id] + _input_ids + [tokenizer.sep_token_id])

            length = len(tokens)
            # piece和word的对应关系
            _pieces2word = np.zeros((length, len(_input_ids)), dtype=np.bool)
            if tokenizer is not None:
                start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(start, start + len(pieces)))
                    _pieces2word[i, pieces[0] + 1: pieces[-1] + 2] = 1
                    start += len(pieces)

            # 相对距离
            _dist_inputs = np.zeros((length, length), dtype=np.int)
            for k in range(length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i, j in itertools.product(range(length), range(length)):
                _dist_inputs[i, j] = DIST_TO_IDX[-_dist_inputs[i, j]] + 9 if _dist_inputs[i, j] < 0 else DIST_TO_IDX[
                    _dist_inputs[i, j]]

            _dist_inputs[_dist_inputs == 0] = 19

            # 标签
            _grid_labels = np.zeros((length, length), dtype=np.int)
            _grid_mask = np.ones((length, length), dtype=np.bool)

            for entity in label:
                if "index" in entity:
                    index = entity["index"]
                else:
                    _start, _end, _type = entity["start_offset"], entity["end_offset"], entity["label"]
                    index = list(range(_start, _end))

                if index[-1] >= max_length - 2:
                    continue

                for i in range(len(index)):
                    if i + 1 >= len(index):
                        break
                    _grid_labels[index[i], index[i + 1]] = 1
                _grid_labels[index[-1], index[0]] = label_to_id[_type] + 2

            for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask, _grid_labels]):
                encoded_inputs[k].append(v)

        return encoded_inputs

    @staticmethod
    def convert_to_features_valid(
        examples: Any,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        text_column_name,
        is_chinese,
    ):

        # 英文文本使用空格分隔单词，BertTokenizer不对空格tokenize
        sentences = list(examples[text_column_name])
        if is_chinese:
            # 将中文文本的空格替换成其他字符，保证标签对齐
            sentences = [text.replace(" ", "-") for text in sentences]

        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask"]
        encoded_inputs = {k: [] for k in input_keys}

        for sentence in sentences:
            tokens = [tokenizer.tokenize(word) for word in sentence[:max_length - 2]]
            pieces = [piece for pieces in tokens for piece in pieces]
            _input_ids = tokenizer.convert_tokens_to_ids(pieces)
            _input_ids = np.array([tokenizer.cls_token_id] + _input_ids + [tokenizer.sep_token_id])

            length = len(tokens)
            # piece和word的对应关系
            _pieces2word = np.zeros((length, len(_input_ids)), dtype=np.bool)
            if tokenizer is not None:
                start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(start, start + len(pieces)))
                    _pieces2word[i, pieces[0] + 1: pieces[-1] + 2] = 1
                    start += len(pieces)

            # 相对距离
            _dist_inputs = np.zeros((length, length), dtype=np.int)
            for k in range(length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i, j in itertools.product(range(length), range(length)):
                _dist_inputs[i, j] = DIST_TO_IDX[-_dist_inputs[i, j]] + 9 if _dist_inputs[i, j] < 0 else DIST_TO_IDX[
                    _dist_inputs[i, j]]

            _dist_inputs[_dist_inputs == 0] = 19
            _grid_mask = np.ones((length, length), dtype=np.bool)

            for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask]):
                encoded_inputs[k].append(v)

        return encoded_inputs

    @property
    def collate_fn(self) -> Optional[Callable]:
        return DataCollatorForW2Ner()

from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from lightningnlp.task.named_entity_recognition.data import TokenClassificationDataModule
from lightningnlp.utils.tensor import sequence_padding


@dataclass
class DataCollatorForCRFNer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None
    ignore_list: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in self.ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple([t[0], int(t[1]), int(t[2]), t[3]]) for t in feature.pop("target")} for
                                   feature in features]
            return batch

        batch_label_ids = torch.zeros_like(batch["input_ids"])
        for i, lb in enumerate(labels):
            for start, end, tag in lb:
                batch_label_ids[i, start] = tag + 1  # B
                batch_label_ids[i, start + 1: end + 1] = tag + self.num_labels + 1  # I

        batch['labels'] = batch_label_ids
        return batch


@dataclass
class DataCollatorForCascadeCRFNer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    ignore_list: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = ([feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None)
        new_features = [{k: v for k, v in f.items() if k not in self.ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple([t[0], int(t[1]), int(t[2]), t[3]]) for t in feature.pop("target")} for
                                   feature in features]
            return batch

        batch_entity_labels = torch.zeros_like(batch["input_ids"])
        batch_entity_ids, batch_labels = [], []
        for i, lb in enumerate(labels):
            entity_ids, label = [], []
            for start, end, tag in lb:
                batch_entity_labels[i, start] = 1  # B
                batch_entity_labels[i, start + 1: end + 1] = 2  # I
                entity_ids.append([start, end])
                label.append(tag + 1)
            if not entity_ids:
                entity_ids.append([0, 0])
                label.append(0)
            batch_entity_ids.append(entity_ids)
            batch_labels.append(label)

        batch['entity_labels'] = batch_entity_labels
        batch['entity_ids'] = torch.from_numpy(sequence_padding(batch_entity_ids))
        batch['labels'] = torch.from_numpy(sequence_padding(batch_labels))
        return batch


class CRFNerDataModule(TokenClassificationDataModule):
    """CRF model for Named Entity Recognition.
    """
    def __init__(self, *args, cascade: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cascade = cascade

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        if self.cascade:
            return DataCollatorForCascadeCRFNer(tokenizer=self.tokenizer, ignore_list=ignore_list)
        return DataCollatorForCRFNer(tokenizer=self.tokenizer, num_labels=len(self.labels), ignore_list=ignore_list)

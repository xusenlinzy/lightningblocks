from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from lightningnlp.task.named_entity_recognition.data import TokenClassificationDataModule


@dataclass
class DataCollatorForLEARNer:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_annotations: Optional[List[str]] = None
    nested: Optional[bool] = False
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

        label_batch = self.tokenizer(
            list(self.label_annotations),
            padding=self.padding,
            truncation=True,
            max_length=64,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        label_batch = {f"label_{k}": v for k, v in label_batch.items()}

        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["texts"] = [feature.pop("text") for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [feature.pop("offset_mapping") for feature in features]
            if "target" in features[0].keys():
                batch['target'] = [{tuple([t[0], int(t[1]), int(t[2]), t[3]]) for t in feature.pop("target")} for
                                   feature in features]
            return {**batch, **label_batch}

        bs, seqlen = batch["input_ids"].shape
        num_labels = len(self.label_annotations)
        batch_start_labels = torch.zeros(bs, seqlen, num_labels, dtype=torch.long)
        batch_end_labels = torch.zeros(bs, seqlen, num_labels, dtype=torch.long)

        batch_span_labels = None
        if self.nested:
            batch_span_labels = torch.zeros(bs, seqlen, seqlen, num_labels, dtype=torch.long)

        for i, lb in enumerate(labels):
            for start, end, tag in lb:
                batch_start_labels[i, start, tag] = 1
                batch_end_labels[i, end, tag] = 1
                if self.nested:
                    batch_span_labels[i, start, end, tag] = 1

        batch["start_labels"] = batch_start_labels
        batch["end_labels"] = batch_end_labels
        if self.nested:
            batch["span_labels"] = batch_span_labels
        return {**batch, **label_batch}


class LEARNerDataModule(TokenClassificationDataModule):
    """LEAR model for Named Entity Recognition.
    """
    def __init__(self, schema2prompt, *args, nested: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.schema2prompt = schema2prompt
        self.nested = nested

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        label_annotations = [self.schema2prompt.get(label) for label in self.labels]
        return DataCollatorForLEARNer(tokenizer=self.tokenizer, label_annotations=label_annotations,
                                      nested=self.nested, ignore_list=ignore_list)

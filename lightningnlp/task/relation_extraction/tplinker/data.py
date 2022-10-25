import torch
from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from lightningnlp.task.relation_extraction.data import RelationExtractionDataModule


@dataclass
class DataCollatorForTPLinkerPlus:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None
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
                batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
            return batch

        bs, seqlen = batch["input_ids"].shape
        mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=0).bool()
        num_tag = self.num_predicates * 4 + 1
        batch_shaking_tag = torch.zeros(bs, seqlen, seqlen, num_tag, dtype=torch.long)

        for i, lb in enumerate(labels):
            for sh, st, p, oh, ot in lb:
                # SH2OH
                batch_shaking_tag[i, sh, oh, p] = 1
                # OH2SH
                batch_shaking_tag[i, oh, sh, p + self.num_predicates] = 1
                # ST2OT
                batch_shaking_tag[i, st, ot, p + self.num_predicates * 2] = 1
                # OT2ST
                batch_shaking_tag[i, ot, st, p + self.num_predicates * 3] = 1
                # EH2ET
                batch_shaking_tag[i, sh, st, -1] = 1
                batch_shaking_tag[i, oh, ot, -1] = 1

        batch["labels"] = batch_shaking_tag.masked_select(mask[None, :, :, None]).reshape(bs, -1, num_tag)
        return batch


class TPlinkerNerDataModule(RelationExtractionDataModule):
    """TPLinker model for Relation Extraction.
    """

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForTPLinkerPlus(tokenizer=self.tokenizer, num_predicates=len(self.predicates), ignore_list=ignore_list)

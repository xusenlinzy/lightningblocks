import torch
from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from lightningnlp.task.relation_extraction.data import RelationExtractionDataModule


@dataclass
class DataCollatorForGPLinker:
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

        bs = batch["input_ids"].size(0)
        max_spo_num = max(len(lb) for lb in labels)
        batch_entity_labels = torch.zeros(bs, 2, max_spo_num, 2, dtype=torch.long)
        batch_head_labels = torch.zeros(bs, self.num_predicates, max_spo_num, 2, dtype=torch.long)
        batch_tail_labels = torch.zeros(bs, self.num_predicates, max_spo_num, 2, dtype=torch.long)

        for i, lb in enumerate(labels):
            for spidx, (sh, st, p, oh, ot) in enumerate(lb):
                batch_entity_labels[i, 0, spidx, :] = torch.tensor([sh, st])
                batch_entity_labels[i, 1, spidx, :] = torch.tensor([oh, ot])
                batch_head_labels[i, p, spidx, :] = torch.tensor([sh, oh])
                batch_tail_labels[i, p, spidx, :] = torch.tensor([st, ot])

        batch["entity_labels"] = batch_entity_labels
        batch["head_labels"] = batch_head_labels
        batch["tail_labels"] = batch_tail_labels
        return batch


class GPLinkerDataModule(RelationExtractionDataModule):
    """GPLinker for Relation Extraction.
    """

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForGPLinker(tokenizer=self.tokenizer, num_predicates=len(self.predicates), ignore_list=ignore_list)

import torch
import random
from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from lightningblocks.task.relation_extraction.data import RelationExtractionDataModule


@dataclass
class DataCollatorForCasRel:
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
        batch_subject_labels = torch.zeros(bs, seqlen, 2, dtype=torch.long)
        batch_object_labels = torch.zeros(bs, seqlen, self.num_predicates, 2, dtype=torch.long)
        batch_subject_ids = torch.zeros(bs, 2, dtype=torch.long)

        for i, lb in enumerate(labels):
            spoes = {}
            for sh, st, p, oh, ot in lb:
                if (sh, st) not in spoes:
                    spoes[(sh, st)] = []
                spoes[(sh, st)].append((oh, ot, p))
            if spoes:
                for s in spoes:
                    batch_subject_labels[i, s[0], 0] = 1
                    batch_subject_labels[i, s[1], 1] = 1
                # 随机选一个subject
                subject_ids = random.choice(list(spoes.keys()))
                batch_subject_ids[i, 0] = subject_ids[0]
                batch_subject_ids[i, 1] = subject_ids[1]
                for o in spoes.get(subject_ids, []):
                    batch_object_labels[i, o[0], o[2], 0] = 1
                    batch_object_labels[i, o[1], o[2], 1] = 1

        batch["subject_labels"] = batch_subject_labels
        batch["object_labels"] = batch_object_labels
        batch["subject_ids"] = batch_subject_ids
        return batch


class CasRelDataModule(RelationExtractionDataModule):
    """CasRel model for Relation Extraction.
    """

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForCasRel(tokenizer=self.tokenizer, num_predicates=len(self.predicates), ignore_list=ignore_list)

import torch
import random
from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from lightningnlp.task.relation_extraction.data import RelationExtractionDataModule


@dataclass
class DataCollatorForPRGC:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None
    negative_ratio: Optional[int] = None
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

        new_batch = {
            "input_ids": [],
            "attention_mask": [],
            "seq_labels": [],
            "corres_labels": [],
            "potential_rels": [],
            "rel_labels": []
        }

        seqlen = batch["input_ids"].shape[1]
        for i, lb in enumerate(labels):
            corres_label = torch.zeros(seqlen, seqlen, dtype=torch.long)
            spoes = {}
            for sh, st, p, oh, ot in lb:
                corres_label[sh, oh] = 1
                if p not in spoes:
                    spoes[p] = []
                spoes[p].append((sh, st, oh, ot))

            # rel one-hot label
            rel_label = torch.zeros(self.num_predicates, dtype=torch.long)
            for p in spoes:
                rel_label[p] = 1

            # positive samples
            for p in spoes:
                # subject, object B-I-O label
                seq_label = torch.zeros(2, seqlen, dtype=torch.long)
                for sh, st, oh, ot in spoes[p]:
                    seq_label[0, sh] = 1  # B-ENT
                    seq_label[0, sh + 1: st + 1] = 2  # I-ENT
                    seq_label[1, oh] = 1  # B-ENT
                    seq_label[1, oh + 1: ot + 1] = 2  # I-ENT
                new_batch["input_ids"].append(batch["input_ids"][i])
                new_batch["attention_mask"].append(batch["attention_mask"][i])
                new_batch["rel_labels"].append(rel_label)
                new_batch["seq_labels"].append(seq_label)
                new_batch["corres_labels"].append(corres_label)
                new_batch["potential_rels"].append(p)

            # negtive samples
            neg_rels = set(range(self.num_predicates)).difference(set(spoes.keys()))
            if neg_rels:
                neg_rels = random.sample(neg_rels, k=min(len(neg_rels), self.negative_ratio))
            for neg_rel in neg_rels:
                # subject, object B-I-O label
                seq_label = torch.zeros(2, seqlen, dtype=torch.long)
                new_batch["input_ids"].append(batch["input_ids"][i])
                new_batch["attention_mask"].append(batch["attention_mask"][i])
                new_batch["rel_labels"].append(rel_label)
                new_batch["seq_labels"].append(seq_label)
                new_batch["corres_labels"].append(corres_label)
                new_batch["potential_rels"].append(neg_rel)

        return {k: torch.stack(v) for k, v in new_batch.items()}


class PRGCDataModule(RelationExtractionDataModule):
    """PRGC model for Relation Extraction.
    """

    def __init__(self, *args, negative_ratio: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.negative_ratio = negative_ratio

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForPRGC(tokenizer=self.tokenizer, num_predicates=len(self.predicates),
                                   ignore_list=ignore_list, negative_ratio=self.negative_ratio)

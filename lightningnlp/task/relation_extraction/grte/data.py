import torch
from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from lightningnlp.task.relation_extraction.data import RelationExtractionDataModule


@dataclass
class DataCollatorForGRTE:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_predicates: Optional[int] = None
    label2id: Optional[dict] = None
    ignore_list: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # sourcery skip: low-code-quality
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
        batch_labels = torch.zeros(bs, seqlen, seqlen, self.num_predicates, dtype=torch.long)
        if self.label2id is None:
            tags = ["N/A", "SS", "MSH", "MST", "SMH", "SMT", "MMH", "MMT"]
            self.label2id = {t: idx for idx, t in enumerate(tags)}

        for i, lb in enumerate(labels):
            spoes = {}
            for sh, st, p, oh, ot in lb:
                if (sh, st) not in spoes:
                    spoes[(sh, st)] = []
                spoes[(sh, st)].append((oh, ot, p))
            if spoes:
                for s in spoes:
                    sh, st = s
                    for oh, ot, p in spoes[(sh, st)]:
                        if sh == st and oh == ot:
                            batch_labels[i, sh, oh, p] = self.label2id['SS']
                        elif sh != st and oh == ot:
                            batch_labels[i, sh, oh, p] = self.label2id['MSH']
                            batch_labels[i, st, oh, p] = self.label2id['MST']
                        elif sh == st:
                            batch_labels[i, sh, oh, p] = self.label2id['SMH']
                            batch_labels[i, sh, ot, p] = self.label2id['SMT']
                        else:
                            batch_labels[i, sh, oh, p] = self.label2id['MMH']
                            batch_labels[i, st, ot, p] = self.label2id['MMT']

        batch["labels"] = batch_labels
        return batch


class GRTEDataModule(RelationExtractionDataModule):
    """GRTE model for Relation Extraction.
    """

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target"]
        return DataCollatorForGRTE(tokenizer=self.tokenizer, num_predicates=len(self.predicates), ignore_list=ignore_list)

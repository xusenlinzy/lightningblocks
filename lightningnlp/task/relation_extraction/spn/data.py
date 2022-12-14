from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Any, Dict

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from ..data import RelationExtractionDataModule


@dataclass
class DataCollatorForSPN:
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
                batch['target'] = [{tuple(t) for t in feature.pop("target")} for feature in features]
            return batch

        spn_labels = []
        for lb in labels:
            spn_label = {
                "relation": [],
                "head_start_index": [],
                "head_end_index": [],
                "tail_start_index": [],
                "tail_end_index": []
            }
            for sh, st, p, oh, ot in lb:
                spn_label["relation"].append(p)
                spn_label["head_start_index"].append(sh)
                spn_label["head_end_index"].append(st)
                spn_label["tail_start_index"].append(oh)
                spn_label["tail_end_index"].append(ot)
            spn_labels.append({k: torch.tensor(v, dtype=torch.long) for k, v in spn_label.items()})

        batch['spn_labels'] = spn_labels
        return batch


class SPNDataModule(RelationExtractionDataModule):
    """Spn model for Relation Extraction.
    """

    @property
    def collate_fn(self) -> Optional[Callable]:
        ignore_list = ["offset_mapping", "text", "target", "spn_labels"]
        return DataCollatorForSPN(tokenizer=self.tokenizer, ignore_list=ignore_list)

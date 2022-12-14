from ..casrel import DataCollatorForCasRel
from ..gplinker import DataCollatorForGPLinker
from ..grte import DataCollatorForGRTE
from ..pfn import DataCollatorForPFN
from ..prgc import DataCollatorForPRGC
from ..spn import DataCollatorForSPN
from ..tplinker import DataCollatorForTPLinkerPlus

RE_COLLATOR_MAP = {
    "casrel": DataCollatorForCasRel,
    "gplinker": DataCollatorForGPLinker,
    "tplinker": DataCollatorForTPLinkerPlus,
    "grte": DataCollatorForGRTE,
    "spn": DataCollatorForSPN,
    "pfn": DataCollatorForPFN,
    "prgc": DataCollatorForPRGC,
}


def get_auto_re_collator(model_name: str = "gplinker"):
    try:
        return RE_COLLATOR_MAP[model_name]
    except KeyError as e:
        raise ValueError(f"Model name must in {RE_COLLATOR_MAP.keys()}.") from e

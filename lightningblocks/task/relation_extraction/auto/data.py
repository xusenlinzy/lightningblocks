from lightningblocks.task.relation_extraction.casrel import DataCollatorForCasRel
from lightningblocks.task.relation_extraction.gplinker import DataCollatorForGPLinker
from lightningblocks.task.relation_extraction.tplinker import DataCollatorForTPLinkerPlus
from lightningblocks.task.relation_extraction.grte import DataCollatorForGRTE
from lightningblocks.task.relation_extraction.spn import DataCollatorForSPN
from lightningblocks.task.relation_extraction.pfn import DataCollatorForPFN
from lightningblocks.task.relation_extraction.prgc import DataCollatorForPRGC


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
    return RE_COLLATOR_MAP[model_name]

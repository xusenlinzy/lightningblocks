from lightningnlp.task.relation_extraction.casrel import DataCollatorForCasRel
from lightningnlp.task.relation_extraction.gplinker import DataCollatorForGPLinker
from lightningnlp.task.relation_extraction.grte import DataCollatorForGRTE
from lightningnlp.task.relation_extraction.pfn import DataCollatorForPFN
from lightningnlp.task.relation_extraction.prgc import DataCollatorForPRGC
from lightningnlp.task.relation_extraction.spn import DataCollatorForSPN
from lightningnlp.task.relation_extraction.tplinker import DataCollatorForTPLinkerPlus

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

from ..cnn import DataCollatorForCNNNer
from ..crf import DataCollatorForCRFNer, DataCollatorForCascadeCRFNer
from ..global_pointer import DataCollatorForGlobalPointer
from ..lear import DataCollatorForLEARNer
from ..mrc import DataCollatorForMRCNer
from ..span import DataCollatorForSpanNer
from ..tplinker import DataCollatorForTPLinkerPlusNer
from ..w2ner import DataCollatorForW2Ner

NER_COLLATOR_MAP = {
    "crf": DataCollatorForCRFNer,
    "cascade-crf": DataCollatorForCascadeCRFNer,
    "softmax": DataCollatorForCRFNer,
    "span": DataCollatorForSpanNer,
    "global-pointer": DataCollatorForGlobalPointer,
    "mrc": DataCollatorForMRCNer,
    "tplinker": DataCollatorForTPLinkerPlusNer,
    "lear": DataCollatorForLEARNer,
    "w2ner": DataCollatorForW2Ner,
    "cnn": DataCollatorForCNNNer,
}


def get_auto_ner_collator(model_name: str = "crf"):
    try:
        return NER_COLLATOR_MAP[model_name]
    except KeyError as e:
        raise ValueError(f"Model name must in {NER_COLLATOR_MAP.keys()}.") from e

from lightningblocks.task.named_entity_recognition.crf import DataCollatorForCRFNer
from lightningblocks.task.named_entity_recognition.crf import DataCollatorForCascadeCRFNer
from lightningblocks.task.named_entity_recognition.span import DataCollatorForSpanNer
from lightningblocks.task.named_entity_recognition.global_pointer import DataCollatorForGlobalPointer
from lightningblocks.task.named_entity_recognition.tplinker import DataCollatorForTPLinkerPlusNer
from lightningblocks.task.named_entity_recognition.mrc import DataCollatorForMRCNer
from lightningblocks.task.named_entity_recognition.lear import DataCollatorForLEARNer
from lightningblocks.task.named_entity_recognition.w2ner import DataCollatorForW2Ner
from lightningblocks.task.named_entity_recognition.cnn import DataCollatorForCNNNer


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
    return NER_COLLATOR_MAP[model_name]

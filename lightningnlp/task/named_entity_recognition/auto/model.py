from typing import Optional

from transformers import PreTrainedModel

from lightningnlp.task.named_entity_recognition.cnn import get_auto_cnn_ner_model
from lightningnlp.task.named_entity_recognition.cnn import get_cnn_model_config
from lightningnlp.task.named_entity_recognition.crf import get_auto_cascade_crf_ner_model
from lightningnlp.task.named_entity_recognition.crf import get_auto_crf_ner_model
from lightningnlp.task.named_entity_recognition.crf import get_auto_softmax_ner_model
from lightningnlp.task.named_entity_recognition.crf import get_cascade_crf_ner_model_config
from lightningnlp.task.named_entity_recognition.crf import get_crf_model_config
from lightningnlp.task.named_entity_recognition.crf import get_softmax_model_config
from lightningnlp.task.named_entity_recognition.global_pointer import get_auto_gp_ner_model
from lightningnlp.task.named_entity_recognition.global_pointer import get_global_pointer_model_config
from lightningnlp.task.named_entity_recognition.lear import get_auto_lear_ner_model
from lightningnlp.task.named_entity_recognition.lear import get_lear_model_config
from lightningnlp.task.named_entity_recognition.mrc import get_auto_mrc_ner_model
from lightningnlp.task.named_entity_recognition.mrc import get_mrc_model_config
from lightningnlp.task.named_entity_recognition.span import get_auto_span_ner_model
from lightningnlp.task.named_entity_recognition.span import get_span_model_config
from lightningnlp.task.named_entity_recognition.tplinker import get_auto_tplinker_ner_model
from lightningnlp.task.named_entity_recognition.tplinker import get_tplinker_model_config
from lightningnlp.task.named_entity_recognition.w2ner import get_auto_w2ner_ner_model
from lightningnlp.task.named_entity_recognition.w2ner import get_w2ner_model_config

NER_MODEL_MAP = {
    "crf": (get_auto_crf_ner_model, get_crf_model_config),
    "cascade-crf": (get_auto_cascade_crf_ner_model, get_cascade_crf_ner_model_config),
    "softmax": (get_auto_softmax_ner_model, get_softmax_model_config),
    "span": (get_auto_span_ner_model, get_span_model_config),
    "global-pointer": (get_auto_gp_ner_model, get_global_pointer_model_config),
    "mrc": (get_auto_mrc_ner_model, get_mrc_model_config),
    "tplinker": (get_auto_tplinker_ner_model, get_tplinker_model_config),
    "lear": (get_auto_lear_ner_model, get_lear_model_config),
    "w2ner": (get_auto_w2ner_ner_model, get_w2ner_model_config),
    "cnn": (get_auto_cnn_ner_model, get_cnn_model_config),
}


def get_auto_ner_model(
    model_name: str = "crf",
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    return NER_MODEL_MAP[model_name][0](model_type, output_attentions, output_hidden_states)


def get_auto_ner_model_config(labels, model_name: str = "crf", **kwargs):
    return NER_MODEL_MAP[model_name][1](labels, **kwargs)

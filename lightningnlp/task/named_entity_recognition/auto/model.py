from typing import Optional

from transformers import PreTrainedModel

from ..cnn import get_auto_cnn_ner_model, get_cnn_model_config
from ..crf import (
    get_auto_cascade_crf_ner_model,
    get_auto_crf_ner_model,
    get_auto_softmax_ner_model,
    get_cascade_crf_ner_model_config,
    get_crf_model_config,
    get_softmax_model_config,
)
from ..global_pointer import get_auto_gp_ner_model, get_global_pointer_model_config
from ..lear import get_auto_lear_ner_model, get_lear_model_config
from ..mrc import get_auto_mrc_ner_model, get_mrc_model_config
from ..span import get_auto_span_ner_model, get_span_model_config
from ..tplinker import get_auto_tplinker_ner_model, get_tplinker_model_config
from ..w2ner import get_auto_w2ner_ner_model, get_w2ner_model_config

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
    try:
        return NER_MODEL_MAP[model_name][0](model_type, output_attentions, output_hidden_states)
    except KeyError as e:
        raise ValueError(f"Model name must in {NER_MODEL_MAP.keys()}.") from e


def get_auto_ner_model_config(labels, model_name: str = "crf", **kwargs):
    try:
        return NER_MODEL_MAP[model_name][1](labels, **kwargs)
    except KeyError as e:
        raise ValueError(f"Model name must in {NER_MODEL_MAP.keys()}.") from e

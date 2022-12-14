from typing import Optional

from transformers import PreTrainedModel

from ..casrel import get_auto_casrel_re_model, get_casrel_model_config
from ..gplinker import get_auto_gplinker_re_model, get_gplinker_model_config
from ..grte import get_auto_grte_re_model, get_grte_model_config
from ..pfn import get_auto_pfn_re_model, get_pfn_model_config
from ..prgc import get_auto_prgc_re_model, get_prgc_model_config
from ..spn import get_auto_spn_re_model, get_spn_model_config
from ..tplinker import get_auto_tplinker_re_model, get_tplinker_model_config

RE_MODEL_MAP = {
    "casrel": (get_auto_casrel_re_model, get_casrel_model_config),
    "gplinker": (get_auto_gplinker_re_model, get_gplinker_model_config),
    "tplinker": (get_auto_tplinker_re_model, get_tplinker_model_config),
    "grte": (get_auto_grte_re_model, get_grte_model_config),
    "spn": (get_auto_spn_re_model, get_spn_model_config),
    "pfn": (get_auto_pfn_re_model, get_pfn_model_config),
    "prgc": (get_auto_prgc_re_model, get_prgc_model_config),
}


def get_auto_re_model(
    model_name: str = "gplinker",
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:
    try:
        return RE_MODEL_MAP[model_name][0](model_type, output_attentions, output_hidden_states)
    except KeyError as e:
        raise ValueError(f"Model name must in {RE_MODEL_MAP.keys()}.") from e


def get_auto_re_model_config(predicates, model_name: str = "gplinker", **kwargs):
    try:
        return RE_MODEL_MAP[model_name][1](predicates, **kwargs)
    except KeyError as e:
        raise ValueError(f"Model name must in {RE_MODEL_MAP.keys()}") from e

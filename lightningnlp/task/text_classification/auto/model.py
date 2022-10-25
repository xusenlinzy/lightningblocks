from typing import Optional
from transformers import PreTrainedModel
from lightningnlp.task.text_classification.fc import get_fc_model_config
from lightningnlp.task.text_classification.fc import get_auto_fc_tc_model
from lightningnlp.task.text_classification.mdp import get_mdp_model_config
from lightningnlp.task.text_classification.mdp import get_auto_mdp_tc_model


TC_MODEL_MAP = {
    "fc": (get_auto_fc_tc_model, get_fc_model_config),
    "mdp": (get_auto_mdp_tc_model, get_mdp_model_config),
}


def get_auto_tc_model(
    model_name: str = "fc",
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    return TC_MODEL_MAP[model_name][0](model_type, output_attentions, output_hidden_states)


def get_auto_tc_model_config(labels, model_name: str = "fc", **kwargs):
    return TC_MODEL_MAP[model_name][1](labels, **kwargs)

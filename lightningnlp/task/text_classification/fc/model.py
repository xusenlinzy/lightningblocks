import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from lightningnlp.task.utils import MODEL_MAP
from lightningnlp.layers.pooling import Pooler


def get_auto_fc_tc_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model = MODEL_MAP[model_type]
    
    class SequenceClassification(parent_model):
        """
        基于BERT的文本分类模型

        Args:
            config: 模型的配置对象
        """
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.num_labels = config.num_labels

            self.pooler_type = getattr(config, 'pooler_type', 'cls')
            if self.pooler_type != "cls":
                self.config.output_hidden_states = True

            self.backbone = base_model(self.config)
            
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            
            self.dropout = nn.Dropout(classifier_dropout)
            self.pooling = Pooler(self.pooler_type)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self, 
            input_ids: Optional[torch.Tensor] = None, 
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> SequenceClassifierOutput:

            outputs = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=self.config.output_hidden_states or output_hidden_states,
            )

            pooled_output = self.dropout(self.pooling(outputs, attention_mask))
            logits = self.classifier(pooled_output)

            loss = self.compute_loss([logits, labels]) if labels is not None else None
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def compute_loss(self, inputs):
            logits, labels = inputs[:2]
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    return SequenceClassification


def get_fc_model_config(label_map, **kwargs):
    model_config = {
        "num_labels": len(label_map), "pooler_type": "cls", "classifier_dropout": 0.3, "label_map": label_map,
    }
    model_config.update(kwargs)
    return model_config

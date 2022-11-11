from typing import Optional, List, Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from lightningnlp.losses import SpanLoss
from lightningnlp.task.utils import SpanOutput, MODEL_MAP
from lightningnlp.utils.tensor import tensor_to_cpu


def get_auto_span_ner_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:
    base_model, parent_model = MODEL_MAP[model_type]

    class SpanForNer(parent_model):
        """
        基于`BERT`的`Span`实体识别模型
        1. 对于每个`token`分别进行对应实体类型的起始位置判断
        2. 分类数目为实体类型数目+1（非实体）

        Args:
            `config`: 模型的配置对象
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.backbone = base_model(config, add_pooling_layer=False)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.start_fc = nn.Linear(config.hidden_size, config.num_labels)
            self.end_fc = nn.Linear(config.hidden_size, config.num_labels)
            self.loss_type = getattr(config, 'loss_type', 'cross_entropy')

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            offset_mapping: Optional[List[Any]] = None,
            target: Optional[List[Any]] = None,
            return_decoded_labels: Optional[bool] = True,
        ) -> SpanOutput:

            outputs = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            sequence_output = self.dropout(outputs[0])

            start_logits = self.start_fc(sequence_output)
            end_logits = self.end_fc(sequence_output)

            loss, predictions = None, None
            if start_positions is not None and end_positions is not None:
                loss = self.compute_loss([
                    start_logits, end_logits, start_positions, end_positions, attention_mask
                ])

            if not self.training and return_decoded_labels:  # 训练时无需解码
                predictions = self.decode(
                    start_logits, end_logits, attention_mask, texts, offset_mapping
                )

            return SpanOutput(
                loss=loss,
                start_logits=start_logits,
                end_logits=end_logits,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, start_logits, end_logits, sequence_mask, texts, offset_mapping):
            start_labels, end_labels = torch.argmax(start_logits, -1), torch.argmax(end_logits, -1)
            start_labels, end_labels = tensor_to_cpu(start_labels), tensor_to_cpu(end_labels)
            id2label = {int(v): k for k, v in self.config.label2id.items()}

            decode_labels = []
            seqlens = tensor_to_cpu(sequence_mask.sum(1))
            for _starts, _ends, l, text, mapping in zip(start_labels, end_labels, seqlens, texts, offset_mapping):
                l = l.item()
                decode_label = set()
                for i, s in enumerate(_starts):
                    s = s.item()
                    if s == 0 or i >= l - 1 or i == 0:
                        continue
                    for j, e in enumerate(_ends[i:]):
                        e = e.item()
                        if i + j >= l - 1:
                            continue
                        if s == e:
                            _start, _end = mapping[i][0], mapping[i + j][1]
                            decode_label.add((
                                id2label[s],
                                _start,
                                _end,
                                text[_start: _end])
                            )
                            break
                decode_labels.append(decode_label)
            return decode_labels

        def compute_loss(self, inputs):
            start_logits, end_logits, start_positions, end_positions, masks = inputs[:5]
            loss_fct = SpanLoss(self.config.num_labels, loss_type=self.loss_type)
            return loss_fct(preds=(start_logits, end_logits), target=(start_positions, end_positions), masks=masks)

    return SpanForNer


def get_span_model_config(labels, **kwargs):
    span_labels = ['O'] + labels
    label2id = {v: i for i, v in enumerate(span_labels)}
    model_config = {
        "num_labels": len(span_labels), "label2id": label2id, "loss_type": "cross_entropy",
    }
    model_config.update(kwargs)
    return model_config

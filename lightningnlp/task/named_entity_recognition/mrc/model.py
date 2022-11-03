import torch
import torch.nn as nn
from typing import Optional, List, Any
from transformers import PreTrainedModel
from lightningnlp.utils.tensor import tensor_to_cpu
from lightningnlp.task.utils import SpanOutput, MODEL_MAP
from lightningnlp.losses.span_loss import SpanLoss
from lightningnlp.layers.layer_norm import LayerNorm


def get_auto_mrc_ner_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model = MODEL_MAP[model_type]

    class MRCForNer(parent_model):
        """
        基于`BERT`的`MRC`实体识别模型
        + 📖 模型的整体思路将实体识别问题转化为每个实体类型下的`spn`预测问题
        + 📖 模型的输入为：原始的待抽取文本和所有标签对应的文本描述（先验知识）进行拼接
        + 📖 简化版本删除了`spn matrix`的预测
        
        Args:
            `config`: 模型的配置对象
        
        Reference:
            ⭐️ [A Unified MRC Framework for Named Entity Recognition.](https://aclanthology.org/2020.acl-main.519.pdf)
            🚀 [Code](https://github.com/ShannonAI/mrc-for-flat-nested-ner)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.backbone = base_model(config, add_pooling_layer=False)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.mid_linear = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.hidden_dropout_prob)
            )
            self.start_fc = nn.Linear(config.hidden_size, 2)
            self.end_fc = nn.Linear(config.hidden_size, 2)

            if config.use_label_embed:
                embed_dims = getattr(config, 'predicate_embed_dims', config.hidden_size)
                self.label_embedding = nn.Embedding(config.num_labels, embed_dims)
                self.cln = LayerNorm(config.hidden_size, conditional_size=embed_dims)

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

            if self.config.use_label_embed:
                batch_size = input_ids.shape[0]
                labels = torch.cat(
                    [torch.arange(self.config.num_labels) for _ in range(batch_size // self.config.num_labels)]).to(
                    input_ids.device)
                label_features = self.label_embedding(labels)
                sequence_output = self.cln([sequence_output, label_features])

            sequence_output = self.mid_linear(sequence_output)
            start_logits = self.start_fc(sequence_output)
            end_logits = self.end_fc(sequence_output)

            loss, predictions = None, None
            if start_positions is not None and end_positions is not None:
                loss = self.compute_loss([
                    start_logits, end_logits, start_positions, end_positions, token_type_ids
                ])

            if not self.training and return_decoded_labels:  # 训练时无需解码
                predictions = self.decode(
                    start_logits, end_logits, token_type_ids,
                    attention_mask, texts, offset_mapping
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

        def decode(self, start_logits, end_logits, token_type_ids, attention_mask, texts, offset_mapping):
            decoded_labels = []
            batch_size = start_logits.shape[0]
            label_list = list(self.config.label_list.keys()) if isinstance(self.config.label_list,
                                                                           dict) else self.config.label_list
            entity_types = label_list * (batch_size // self.config.num_labels)

            starts = torch.argmax(start_logits, -1) * token_type_ids
            ends = torch.argmax(end_logits, -1) * token_type_ids

            starts, ends = tensor_to_cpu(starts), tensor_to_cpu(ends)
            seqlens = tensor_to_cpu(attention_mask.sum(1))

            for _starts, _ends, l, text, mapping, _type in zip(starts, ends, seqlens, texts, offset_mapping, entity_types):
                entities = set()
                l = l.item()
                for i, s in enumerate(_starts):
                    s = s.item()
                    if s == 0 or i >= l - 1:
                        continue
                    for j, e in enumerate(_ends[i:]):
                        e = e.item()
                        if i + j >= l - 1:
                            continue
                        if s == e:
                            _start, _end = mapping[i][0], mapping[i + j][1]
                            entities.add((
                                _type,
                                _start,
                                _end,
                                text[_start: _end])
                            )
                            break
                decoded_labels.append(entities)

            return [set.union(*decoded_labels[i: i + self.config.num_labels]) for i in
                    range(0, batch_size, self.config.num_labels)]

        def compute_loss(self, inputs):
            start_logits, end_logits, start_positions, end_positions, masks = inputs[:5]
            loss_fct = SpanLoss(self.config.num_labels, loss_type=self.loss_type)
            return loss_fct(preds=(start_logits, end_logits), target=(start_positions, end_positions), masks=masks)

    return MRCForNer


def get_mrc_model_config(labels, **kwargs):
    model_config = {
        "num_labels": len(labels), "label_list": labels, "use_label_embed": False, "loss_type": "cross_entropy",
    }
    model_config.update(kwargs)
    return model_config

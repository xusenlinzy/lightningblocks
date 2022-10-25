import torch
import torch.nn as nn
from typing import Optional, List, Any
from transformers import PreTrainedModel
from lightningblocks.utils.tensor import tensor_to_cpu
from lightningblocks.task.utils import SpanOutput, MODEL_MAP
from lightningblocks.losses.span_loss import SpanLossForMultiLabel
from lightningblocks.layers.lear import LabelFusionForToken, Classifier, MLPForMultiLabel


def get_auto_lear_ner_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model = MODEL_MAP[model_type]

    class LEARForNer(parent_model):
        """
        基于`BERT`的`LEAR`实体识别模型
        + 📖 模型的整体思路将实体识别问题转化为每个实体类型下的`spn`预测问题
        + 📖 模型的输入分为两个部分：原始的待抽取文本和所有标签对应的文本描述（先验知识）
        + 📖 原始文本和标签描述文本共享`BERT`的编码器权重
        + 📖 采用注意力机制融合标签信息到`token`特征中去
        
        Args:
            `config`: 模型的配置对象
        
        Reference:
            ⭐️ [Enhanced Language Representation with Label Knowledge for Span Extraction.](https://aclanthology.org/2021.emnlp-main.379.pdf)
            🚀 [Code](https://github.com/Akeepers/LEAR)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.bert = base_model(config, add_pooling_layer=False)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            # 将标签信息融合到token的特征当中
            self.label_fusion_layer = LabelFusionForToken(config.hidden_size)
            self.start_classifier = Classifier(config.hidden_size, config.num_labels)
            self.end_classifier = Classifier(config.hidden_size, config.num_labels)

            # 嵌套NER则增加一个span matrix的预测
            if config.nested:
                self.span_layer = MLPForMultiLabel(config.hidden_size * 2, config.num_labels)

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            label_input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            label_attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            label_token_type_ids: Optional[torch.Tensor] = None,
            start_labels: Optional[torch.Tensor] = None,
            end_labels: Optional[torch.Tensor] = None,
            span_labels: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            offset_mapping: Optional[List[Any]] = None,
            target: Optional[List[Any]] = None,
            return_decoded_labels: Optional[bool] = True,
        ) -> SpanOutput:

            token_features = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )[0]
            token_features = self.dropout(token_features)

            label_features = self.bert(
                label_input_ids,
                attention_mask=label_attention_mask,
                token_type_ids=label_token_type_ids,
            )[0]

            fused_features = self.label_fusion_layer(token_features, label_features, label_attention_mask)

            start_logits = self.start_classifier(fused_features)
            end_logits = self.end_classifier(fused_features)

            span_logits = None
            if self.config.nested:
                seqlen = input_ids.shape[1]
                start_extend = fused_features.unsqueeze(2).expand(-1, -1, seqlen, -1, -1)
                end_extend = fused_features.unsqueeze(1).expand(-1, seqlen, -1, -1, -1)
                span_matrix = torch.cat((start_extend, end_extend), dim=-1)
                span_logits = self.span_layer(span_matrix)

            loss, predictions = None, None
            if start_labels is not None and end_labels is not None:
                if self.config.nested:
                    loss = self.compute_loss([
                        start_logits, end_logits, span_logits, start_labels, end_labels, span_labels, attention_mask
                    ])
                else:
                    loss = self.compute_loss([
                        start_logits, end_logits, start_labels, end_labels, attention_mask
                    ])

            if not self.training and return_decoded_labels:  # 训练时无需解码
                predictions = self.decode(start_logits, end_logits, span_logits, attention_mask, texts, offset_mapping)

            return SpanOutput(
                loss=loss,
                start_logits=start_logits,
                end_logits=end_logits,
                span_logits=span_logits,
                predictions=predictions,
                groundtruths=target)

        def decode(self, start_logits, end_logits, span_logits, attention_mask, texts, offset_mapping):
            decode_labels = []

            if not self.config.nested:
                seqlens = tensor_to_cpu(attention_mask.sum(1))
                start_preds, end_preds = torch.sigmoid(start_logits), torch.sigmoid(end_logits)
                start_preds, end_preds = tensor_to_cpu(start_preds), tensor_to_cpu(end_preds)

                start_thresh = getattr(self.config, "start_thresh", 0.5)
                end_thresh = getattr(self.config, "end_thresh", 0.5)
                id2label = {int(v): k for k, v in self.config.label2id.items()}

                for starts, ends, l, text, mapping in zip(start_preds, end_preds, seqlens, texts, offset_mapping):
                    l = l.item()
                    decode_label = set()
                    starts, ends = torch.where(starts > start_thresh), torch.where(ends > end_thresh)
                    for _start, c1 in zip(*starts):
                        _start, c1 = _start.item(), c1.item()
                        if _start == 0 or _start >= l - 1:
                            continue
                        for _end, c2 in zip(*ends):
                            _end, c2 = _end.item(), c2.item()
                            if _start <= _end < l - 1 and c1 == c2:
                                s, e = mapping[_start][0], mapping[_end][1]
                                decode_label.add((
                                    id2label[c1], s, e, text[s: e]
                                ))
                                break  # 就近原则
                    decode_labels.append(decode_label)

                return decode_labels

            bs, seq_len, num_labels = start_logits.shape
            masks = attention_mask.unsqueeze(-1).expand(-1, -1, num_labels)
            start_label_masks = masks.unsqueeze(-2).expand(-1, -1, seq_len, -1).bool()
            end_label_masks = masks.unsqueeze(-3).expand(-1, seq_len, -1, -1).bool()

            # [batch_size, seq_len, num_labels]
            start_preds, end_preds = (start_logits > 0).bool, (end_logits > 0).bool()
            # [batch_size, seq_len, seq_len, num_labels]
            match_preds = span_logits > 0

            match_preds = (
                    match_preds & start_preds.unsqueeze(2).expand(-1, -1, seq_len, -1)
                    & end_preds.unsqueeze(1).expand(-1, seq_len, -1, -1)
            )

            # noinspection PyUnresolvedReferences
            match_label_masks = torch.triu((start_label_masks & end_label_masks, 0).permute(0, 3, 1, 2), 0)
            match_label_masks = match_label_masks.permute(0, 2, 3, 1)
            match_preds = match_preds & match_label_masks

            for preds, text, mapping in zip(match_preds, texts, offset_mapping):
                decode_label = set()
                preds = torch.where(preds == True)
                for start_idx, end_idx, label_id in preds:
                    start_idx, end_idx, label_id = start_idx.item(), end_idx.item(), label_id.item()
                    _start, _end = mapping[start_idx][0].item(), mapping[end_idx][1].item()
                    decode_label.add((
                        id2label[label_id], _start, _end, text[_start: _end]
                    ))
            return decode_labels

        def compute_loss(self, inputs):
            loss_fct = SpanLossForMultiLabel()

            if not self.config.nested:
                start_logits, end_logits, start_labels, end_labels, mask = inputs[:5]
                loss = loss_fct(
                    (start_logits, end_logits),
                    (start_labels, end_labels), mask
                )
                return loss
            start_logits, end_logits, span_logits, start_labels, end_labels, span_labels, mask = inputs[:7]
            loss = loss_fct(
                (start_logits, end_logits, span_logits),
                (start_labels, end_labels, span_labels), mask, nested=True
            )
            return loss

    return LEARForNer


def get_lear_model_config(labels, **kwargs):
    label2id = {v: i for i, v in enumerate(labels)}
    model_config = {
        "num_labels": len(labels), "label2id": label2id,
        "nested": False, "start_thresh": 0.5, "end_thresh": 0.5,
    }
    model_config.update(kwargs)
    return model_config

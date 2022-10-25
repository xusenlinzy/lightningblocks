import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Any
from transformers import PreTrainedModel
from lightningnlp.utils.tensor import tensor_to_cpu
from lightningnlp.task.utils import SequenceLabelingOutput, MODEL_MAP
from lightningnlp.losses import MultiLabelCategoricalCrossEntropy
from lightningnlp.losses import SparseMultilabelCategoricalCrossentropy
from lightningnlp.layers.global_pointer import GlobalPointer, EfficientGlobalPointer, Biaffine, UnlabeledEntity


def get_auto_gp_ner_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model = MODEL_MAP[model_type]

    class GlobalPointerForNer(parent_model):
        """
        基于`BERT`的`GlobalPointer`实体识别模型
        + 📖 模型的整体思路将实体识别问题转化为每个实体类型下`token`对之间的二分类问题，用统一的方式处理嵌套和非嵌套`NER`
        + 📖 采用多头注意力得分的计算方式来建模`token`对之间的得分
        + 📖 采用旋转式位置编码加入相对位置信息
        + 📖 采用单目标多分类交叉熵推广形式的多标签分类损失函数解决类别不平衡问题

        Args:
            `config`: 模型的配置对象
        
        Reference:
            ⭐️ [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)
            ⭐️ [Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877) \\
            🚀 [Code](https://github.com/bojone/GlobalPointer)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.bert = base_model(config, add_pooling_layer=False)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            head_type = getattr(config, 'head_type', 'efficient_global_pointer')
            # token对特征的计算方式
            if head_type == "efficient_global_pointer":
                self.global_pointer = EfficientGlobalPointer(config.hidden_size, config.head_size,
                                                             config.num_labels, use_rope=config.use_rope)
            elif head_type == "gplinker":
                self.global_pointer = GlobalPointer(config.hidden_size, config.head_size,
                                                    config.num_labels, use_rope=config.use_rope)
            elif head_type == "biaffine":
                self.global_pointer = Biaffine(config.hidden_size, config.head_size, config.num_labels)

            elif head_type == "unlabeled_entity":
                self.global_pointer = UnlabeledEntity(config.hidden_size, config.num_labels)
                
            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            offset_mapping: Optional[List[Any]] = None,
            target: Optional[List[Any]] = None,
            return_decoded_labels: Optional[bool] = True,
        ) -> SequenceLabelingOutput:

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            sequence_output = self.dropout(outputs[0])
            logits = self.global_pointer(sequence_output, mask=attention_mask)

            loss, predictions = None, None
            if labels is not None:
                sparse = getattr(self.config, 'is_sparse', False)
                loss = self.compute_loss([logits, labels, attention_mask], sparse=sparse)

            if not self.training and return_decoded_labels:  # 训练时无需解码
                predictions = self.decode(logits, attention_mask, texts, offset_mapping)

            return SequenceLabelingOutput(
                loss=loss,
                logits=logits,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, logits, masks, texts, offset_mapping):
            all_entity_list = []
            seq_lens, logits = tensor_to_cpu(masks.sum(1)), tensor_to_cpu(logits).float()
            id2label = {int(v): k for k, v in self.config.label2id.items()}

            decode_thresh = getattr(self.config, "decode_thresh", 0.0)
            for _logits, l, text, mapping in zip(logits, seq_lens, texts, offset_mapping):
                entity_list = set()
                l = l.item()

                for label_id, start_idx, end_idx in zip(*torch.where(_logits > decode_thresh)):
                    label_id, start_idx, end_idx = label_id.item(), start_idx.item(), end_idx.item()
                    if start_idx >= (l - 1) or end_idx >= (l - 1) or 0 in [start_idx, end_idx]:
                        continue
                    label = id2label[label_id]
                    _start, _end = mapping[start_idx][0], mapping[end_idx][1]
                    entity_list.add((label, _start, _end, text[_start: _end]))
                all_entity_list.append(set(entity_list))

            return all_entity_list

        def compute_loss(self, inputs, sparse=True):
            """ 
            便于使用自定义的损失函数
            """
            preds, target = inputs[:2]
            shape = preds.shape
            if not sparse:
                loss_fct = MultiLabelCategoricalCrossEntropy()
                return loss_fct(preds=preds.reshape(shape[0] * self.config.num_labels, -1),
                                target=target.reshape(shape[0] * self.config.num_labels, -1))
            else:
                target = target[..., 0] * shape[2] + target[..., 1]  # [bsz, heads, num_spoes]
                preds = preds.reshape(shape[0], -1, np.prod(shape[2:]))
                loss_fct = SparseMultilabelCategoricalCrossentropy(mask_zero=True)
                return loss_fct(preds, target).sum(dim=1).mean()

    return GlobalPointerForNer


def get_global_pointer_model_config(labels, **kwargs):
    label2id = {v: i for i, v in enumerate(labels)}
    model_config = {
        "num_labels": len(labels), "label2id": label2id,
        "head_size": 64, "use_rope": True, "is_sparse": False,
        "head_type": "efficient_global_pointer", "decode_thresh": 0.,
    }
    model_config.update(kwargs)
    return model_config

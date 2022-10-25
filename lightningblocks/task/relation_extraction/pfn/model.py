import itertools
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Any
from transformers import PreTrainedModel
from lightningblocks.utils.tensor import tensor_to_numpy
from lightningblocks.task.utils import RelationExtractionOutput, MODEL_MAP
from lightningblocks.layers.pfn import encoder, ner_unit, re_unit


def get_auto_pfn_re_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model = MODEL_MAP[model_type]



    class PFN(parent_model):
        """
        基于`BERT`的`PFN`关系抽取模型
        + 📖 一般的联合抽取模型将实体抽取和关系分类分成两步进行，忽略了两个任务之间的联系
        + 📖 该模型通过分组过滤机制，将隐藏状态分成实体抽取信息、关系抽取信息和共享信息三部分
        + 📖 基于实体抽取信息和共享信息抽取出主语和宾语，基于关系抽取信息和共享信息抽取出对应的关系

        Args:
            `config`: 模型的配置

        Reference:
            ⭐️ [A Partition Filter Network for Joint Entity and Relation Extraction.](https://aclanthology.org/2021.emnlp-main.17.pdf)
            🚀 [Code](https://github.com/Coopercoppers/PFN)
        """
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.bert = base_model(config, add_pooling_layer=False)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.pfn_hidden_size = getattr(config, "pfn_hidden_size", 300)
            self.feature_extractor = encoder(self.pfn_hidden_size, config.hidden_size)

            self.ner = ner_unit(self.pfn_hidden_size, 2, classifier_dropout)
            self.re_head = re_unit(self.pfn_hidden_size, config.num_predicates, classifier_dropout)
            self.re_tail = re_unit(self.pfn_hidden_size, config.num_predicates, classifier_dropout)

            self.dropout = nn.Dropout(classifier_dropout)

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            entity_labels: Optional[torch.Tensor] = None,
            head_labels: Optional[torch.Tensor] = None,
            tail_labels: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            offset_mapping: Optional[List[Any]] = None,
            target: Optional[List[Any]] = None,
        ) -> RelationExtractionOutput:

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            sequence_output = self.dropout(outputs[0])  # [batch_size, seq_len, hidden_size]

            sequence_output = sequence_output.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
            mask = attention_mask.transpose(0, 1)  # [seq_len, batch_size]
            h_ner, h_re, h_share = self.feature_extractor(sequence_output)

            ner_score = self.ner(h_ner, h_share, mask).permute(2, 3, 0, 1)
            re_head_score = self.re_head(h_re, h_share, mask).permute(2, 3, 0, 1)
            re_tail_score = self.re_tail(h_share, h_re, mask).permute(2, 3, 0, 1)

            loss, predictions = None, None
            if entity_labels is not None and head_labels is not None and tail_labels is not None:
                entity_loss = self.compute_loss([ner_score, entity_labels])
                head_loss = self.compute_loss([re_head_score, head_labels])
                tail_loss = self.compute_loss([re_tail_score, tail_labels])
                loss = entity_loss + head_loss + tail_loss

            if not self.training:
                predictions = self.decode(ner_score, re_head_score, re_tail_score, attention_mask, texts,
                                          offset_mapping)

            return RelationExtractionOutput(
                loss=loss,
                logits=None,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, ner_score, re_head_score, re_tail_score, attention_mask, texts, offset_mapping):
            ner_score = tensor_to_numpy(ner_score)
            re_head_score = tensor_to_numpy(re_head_score)
            re_tail_score = tensor_to_numpy(re_tail_score)
            masks = tensor_to_numpy(attention_mask)

            batch_size = ner_score.size(0)
            decode_thresh = getattr(self.config, "decode_thresh", 0.5)
            id2predicate = {int(v): k for k, v in self.config.predicate2id.items()}

            all_spo_list = []
            for bs in range(batch_size):
                # 抽取主体和客体
                subjects, objects = set(), set()
                _ner_score, l = ner_score[bs], masks[bs].sum()
                text, mapping = texts[bs], offset_mapping[bs]
                for r, h, t in zip(*np.where(_ner_score > decode_thresh)):
                    if h >= (l - 1) or t >= (l - 1) or 0 in [h, t]:  # 排除[CLS]、[SEP]、[PAD]
                        continue
                    if r == 0:
                        subjects.add((h, t))
                    else:
                        objects.add((h, t))

                # 识别对应的关系类型
                spoes = set()
                _head_logits, _tail_logits = re_head_score[bs], re_tail_score[bs]
                for (sh, st), (oh, ot) in itertools.product(subjects, objects):
                    p1s = np.where(_head_logits[:, sh, oh] > decode_thresh)[0]
                    p2s = np.where(_tail_logits[:, st, ot] > decode_thresh)[0]
                    ps = set(p1s) & set(p2s)
                    for p in ps:
                        spoes.add((
                            id2predicate[p],
                            text[mapping[sh][0]: mapping[st][1]],
                            text[mapping[oh][0]: mapping[ot][1]]
                        ))
                all_spo_list.append(spoes)
            return all_spo_list

        def compute_loss(self, inputs):
            logits, labels = inputs[:2]
            loss_fct = nn.BCELoss(reduction='sum')
            return loss_fct(logits, labels.float()) / logits.size(-1)

    return PFN


def get_pfn_model_config(predicates, **kwargs):
    predicate2id = {v: i for i, v in enumerate(predicates)}

    model_config = {
        "num_predicates": len(predicates), "predicate2id": predicate2id,
        "pfn_hidden_size": 300, "decode_thresh": 0.5,
    }
    model_config.update(kwargs)
    return model_config

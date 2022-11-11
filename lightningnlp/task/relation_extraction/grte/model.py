from typing import Optional, List, Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from lightningnlp.layers.transformer import TransformerDecoderLayer
from lightningnlp.task.utils import RelationExtractionOutput, MODEL_MAP
from lightningnlp.utils.tensor import tensor_to_cpu


def get_auto_grte_re_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model = MODEL_MAP[model_type]

    # noinspection PyUnboundLocalVariable
    class GRTE(parent_model):
        """
        基于`BERT`的`GRTE`关系抽取模型
        + 📖 模型的整体思路将三元组抽取问题转化为一个表格填充问题，对`token pair`进行多分类
        + 📖 根据实体是否由多个`token`组成将`token pair`之间的关系分成八类
        + 📖 主体-客体-首尾（`S`：单，`M`：多，`H`：首，`T`：尾）：`None`、`SS`、`SMH`、`SMT`、`MSH`、`MST`、`MMH`、`MMT`
        + 📖 全局特征采用`transformer`的带交叉注意力的`encoder`进行迭代学习
        + 📖 采用前向、后向解码的方式进行预测

        Args:
            `config`: 模型的配置

        Reference:
            ⭐️ [A Novel Global Feature-Oriented Relational Triple Extraction Model based on Table Filling.](https://aclanthology.org/2021.emnlp-main.208.pdf)
            🚀 [Official Code](https://github.com/neukg/GRTE)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.backbone = base_model(config, add_pooling_layer=False)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.Lr_e1 = nn.Linear(config.hidden_size, config.hidden_size)
            self.Lr_e2 = nn.Linear(config.hidden_size, config.hidden_size)

            self.elu = nn.ELU()
            self.Cr = nn.Linear(config.hidden_size, config.num_predicates * config.num_labels)

            self.Lr_e1_rev = nn.Linear(config.num_predicates * config.num_labels, config.hidden_size)
            self.Lr_e2_rev = nn.Linear(config.num_predicates * config.num_labels, config.hidden_size)
            self.e_layer = TransformerDecoderLayer(config)

            # 正交初始化
            torch.nn.init.orthogonal_(self.Lr_e1.weight, gain=1)
            torch.nn.init.orthogonal_(self.Lr_e2.weight, gain=1)
            torch.nn.init.orthogonal_(self.Cr.weight, gain=1)
            torch.nn.init.orthogonal_(self.Lr_e1_rev.weight, gain=1)
            torch.nn.init.orthogonal_(self.Lr_e2_rev.weight, gain=1)

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            offset_mapping: Optional[List[Any]] = None,
            target: Optional[List[Any]] = None,
        ) -> RelationExtractionOutput:

            outputs = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            sequence_output = self.dropout(outputs[0])  # [batch_size, seq_len, hidden_size]

            bs, l = sequence_output.shape[:2]
            e1 = self.Lr_e1(sequence_output)
            e2 = self.Lr_e2(sequence_output)

            for i in range(self.config.rounds):
                h = self.elu(e1.unsqueeze(2).repeat(1, 1, l, 1) * e2.unsqueeze(1).repeat(1, l, 1, 1))
                table_logist = self.Cr(h)
                if i != self.config.rounds - 1:
                    table_e1 = table_logist.max(dim=2).values
                    table_e2 = table_logist.max(dim=1).values
                    e1_ = self.Lr_e1_rev(table_e1)
                    e2_ = self.Lr_e2_rev(table_e2)

                    e1 = e1 + self.e_layer(e1_, sequence_output, attention_mask)[0]
                    e2 = e2 + self.e_layer(e2_, sequence_output, attention_mask)[0]

            logits = table_logist.reshape([bs, l, l, self.config.num_predicates, self.config.num_labels])

            loss, predictions = None, None
            if labels is not None and attention_mask is not None:
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                label_masks = attention_mask[:, None, :, None] * attention_mask[:, :, None, None]
                label_masks = label_masks.expand(-1, -1, -1, self.config.num_predicates)
                loss = loss_fct(logits.reshape(-1, self.config.num_labels), labels.reshape([-1]).long())
                loss = (loss * label_masks.reshape([-1])).sum()

            if not self.training:  # 训练时无需解码
                predictions = self.decode(logits, attention_mask, texts, offset_mapping)

            return RelationExtractionOutput(
                loss=loss,
                logits=None,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, logits, masks, texts, offset_mapping):
            logits = tensor_to_cpu(logits.argmax(-1))
            seqlens = tensor_to_cpu(masks.sum(1))
            id2predicate = {int(v): k for k, v in self.config.predicate2id.items()}

            triples = []
            for _logits, length, text, mapping in zip(logits, seqlens, texts, offset_mapping):
                tmp = []
                length = length.item()
                for s, e, r in zip(*torch.where(_logits != self.config.label2id["N/A"])):
                    s, e, r = s.item(), e.item(), r.item()
                    if length - 1 <= s or length - 1 <= e or 0 in [s, e]:
                        continue
                    tmp.append((s, e, r))

                spoes = set()
                for s, e, r in tmp:
                    if _logits[s, e, r] == self.config.label2id["SS"]:
                        spoes.add((
                            id2predicate[r],
                            text[mapping[s][0]: mapping[s][1]],
                            text[mapping[e][0]: mapping[e][1]]
                        ))
                    elif _logits[s, e, r] == self.config.label2id["SMH"]:
                        for s_, e_, r_ in tmp:
                            if r == r_ and _logits[s_, e_, r_] == self.config.label2id["SMT"] and s_ == s and e_ > e:
                                spoes.add((
                                    id2predicate[r],
                                    text[mapping[s][0]: mapping[s][1]],
                                    text[mapping[e][0]: mapping[e_][1]]
                                ))
                                break
                    elif _logits[s, e, r] == self.config.label2id["MMH"]:
                        for s_, e_, r_ in tmp:
                            if r == r_ and _logits[s_, e_, r_] == self.config.label2id["MMT"] and s_ > s and e_ > e:
                                spoes.add((
                                    id2predicate[r],
                                    text[mapping[s][0]: mapping[s_][1]],
                                    text[mapping[e][0]: mapping[e_][1]]
                                ))
                                break
                    elif _logits[s, e, r] == self.config.label2id["MSH"]:
                        for s_, e_, r_ in tmp:
                            if r == r_ and _logits[s_, e_, r_] == self.config.label2id["MST"] and s_ > s and e_ == e:
                                spoes.add((
                                    id2predicate[r],
                                    text[mapping[s][0]: mapping[s_][1]],
                                    text[mapping[e][0]: mapping[e][1]]
                                ))
                                break
                triples.append(spoes)
            return triples

    return GRTE


def get_grte_model_config(predicates, **kwargs):
    predicate2id = {v: i for i, v in enumerate(predicates)}

    tags = ["N/A", "SS", "MSH", "MST", "SMH", "SMT", "MMH", "MMT"]
    label2id = {t: idx for idx, t in enumerate(tags)}

    model_config = {
        "num_predicates": len(predicates), "predicate2id": predicate2id,
        "label2id": label2id, "num_labels": len(tags), "rounds": 3,
    }
    model_config.update(kwargs)
    return model_config

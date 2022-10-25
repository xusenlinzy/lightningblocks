import torch
import torch.nn as nn
from typing import Optional, List, Any
from transformers import PreTrainedModel
from lightningnlp.utils.tensor import tensor_to_list
from lightningnlp.task.utils import RelationExtractionOutput, MODEL_MAP
from lightningnlp.losses.set_loss import SetCriterion
from lightningnlp.layers.set_decoder import SetDecoder


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def generate_span(start_logits, end_logits, masks, config):
    seq_lens = tensor_to_list(masks.sum(1))  # including [CLS] and [SEP]
    outputs = []

    start_probs = tensor_to_list(start_logits.softmax(-1))  # [b, n, l]
    end_probs = tensor_to_list(end_logits.softmax(-1))
    for (start_prob, end_prob, seq_len) in zip(start_probs, end_probs, seq_lens):
        output = {}
        for triple_id in range(config.num_generated_triples):
            predictions = []
            start_indexes = _get_best_indexes(start_prob[triple_id], config.n_best_size)
            end_indexes = _get_best_indexes(end_prob[triple_id], config.n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the sentence. We throw out all
                    # invalid predictions.
                    if start_index >= (seq_len - 1):  # [SEP]
                        continue
                    if end_index >= (seq_len - 1):
                        continue
                    if start_index == 0 or end_index == 0:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > config.max_span_length:
                        continue
                    predictions.append((start_index, end_index))
            output[triple_id] = predictions
        outputs.append(output)
    return outputs


def generate_relation(pred_rel_logits, config):
    _, pred_rels = torch.max(pred_rel_logits.softmax(-1), dim=2)
    pred_rels = tensor_to_list(pred_rels)
    outputs = []
    for pred_rel in pred_rels:
        output = {triple_id: pred_rel[triple_id] for triple_id in range(config.num_generated_triples)}

        outputs.append(output)
    return outputs


def generate_strategy(pred_rel, pred_head, pred_tail, text, mapping, num_predicates, id2predicate):
    if pred_rel != num_predicates and pred_head and pred_tail:
        for ele in pred_head:
            if ele[0] != 0:
                break
        # noinspection PyUnboundLocalVariable
        head = ele
        for ele in pred_tail:
            if ele[0] != 0:
                break
        tail = ele
        return (
            id2predicate[pred_rel],
            text[mapping[head[0]][0]: mapping[head[1]][1]],
            text[mapping[tail[0]][0]: mapping[tail[1]][1]]
        )
    else:
        return


def get_auto_spn_re_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:
    base_model, parent_model = MODEL_MAP[model_type]

    class SPN(parent_model):
        """
        基于`BERT`的`SPN`关系抽取模型
        + 📖 模型的整体思路将三元组抽取问题转化为序列到集合的预测问题
        + 📖 采用`encoder-decoder`的架构
        + 📖 `encoder`采用`bert`提取`token`的特征，`decoder`采用非自回归的`transformer decoder`提取三元组特征
        + 📖 `decoder`每个位置只需要预测一个三元组，最后采用二分图匹配损失消除顺序影响

        Args:
            `config`: 模型的配置对象

        Reference:
            ⭐️ [Joint Entity and Relation Extraction with Set Prediction Networks.](http://xxx.itp.ac.cn/pdf/2011.01675v2)
            🚀 [Official Code](https://github.com/DianboWork/SPN4RE)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.bert = base_model(config, add_pooling_layer=False)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.decoder = SetDecoder(config, config.num_generated_triples,
                                      config.num_decoder_layers, config.num_predicates)

            self.criterion = SetCriterion(config.num_predicates,
                                          loss_weight=self.get_loss_weight(config),
                                          na_coef=config.na_rel_coef,
                                          losses=["entity", "relation"],
                                          matcher=config.matcher)

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            spn_labels: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            offset_mapping: Optional[List[Any]] = None,
            target: Optional[List[Any]] = None,
        ) -> RelationExtractionOutput:

            # encoder
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            encoder_hidden_states = self.dropout(outputs[0])

            # decoder
            decoder_outputs = self.decoder(hidden_states=None,
                                           encoder_hidden_states=encoder_hidden_states,
                                           encoder_attention_mask=attention_mask)
            class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = decoder_outputs

            # [bsz, num_generated_triples, seq_len]
            head_start_logits = head_start_logits.masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
            head_end_logits = head_end_logits.masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)

            tail_start_logits = tail_start_logits.masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
            tail_end_logits = tail_end_logits.masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)

            preds = {'pred_rel_logits': class_logits,
                     'head_start_logits': head_start_logits,
                     'head_end_logits': head_end_logits,
                     'tail_start_logits': tail_start_logits,
                     'tail_end_logits': tail_end_logits}

            loss, predictions = None, None
            if spn_labels is not None:
                loss = self.criterion(preds, spn_labels)

            if not self.training:  # 训练时无需解码
                predictions = self.decode(preds, attention_mask, texts, offset_mapping)

            return RelationExtractionOutput(
                loss=loss,
                logits=None,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        @staticmethod
        def get_loss_weight(config):
            return {"relation": config.rel_loss_weight,
                    "head_entity": config.head_ent_loss_weight,
                    "tail_entity": config.tail_ent_loss_weight}

        def decode(self, preds, masks, texts, offset_mapping):
            pred_head_ent_dict = generate_span(preds["head_start_logits"], preds["head_end_logits"], masks, self.config)
            pred_tail_ent_dict = generate_span(preds["tail_start_logits"], preds["tail_end_logits"], masks, self.config)
            pred_rel_dict = generate_relation(preds['pred_rel_logits'], self.config)
            id2predicate = {int(v): k for k, v in self.config.predicate2id.items()}

            triples = []
            for i in range(len(pred_rel_dict)):
                spoes = set()
                text, mapping = texts[i], offset_mapping[i]
                for triple_id in range(self.config.num_generated_triples):
                    pred_rel = pred_rel_dict[i][triple_id]
                    pred_head = pred_head_ent_dict[i][triple_id]
                    pred_tail = pred_tail_ent_dict[i][triple_id]
                    triple = generate_strategy(pred_rel, pred_head, pred_tail, text, mapping, self.config.num_predicates, id2predicate)
                    if triple:
                        spoes.add(triple)
                triples.append(spoes)
            return triples

    return SPN


def get_spn_model_config(predicates, **kwargs):
    predicate2id = {v: i for i, v in enumerate(predicates)}
    model_config = {
        "num_predicates": len(predicates), "predicate2id": predicate2id,
        "num_decoder_layers": 3, "num_generated_triples": 10, "matcher": "avg", "na_rel_coef": 0.25,
        "rel_loss_weight": 1.0, "head_ent_loss_weight": 2.0, "tail_ent_loss_weight": 2.0,
        "max_span_length": 10, "n_best_size": 50,
    }
    model_config.update(kwargs)
    return model_config

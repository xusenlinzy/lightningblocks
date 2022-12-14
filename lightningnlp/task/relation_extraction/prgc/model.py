from typing import Optional, List, Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...utils import RelationExtractionOutput, MODEL_MAP, get_entities
from ....utils.tensor import tensor_to_numpy


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, hidden_size // 2)
        self.hidden2tag = nn.Linear(hidden_size // 2, self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        return self.hidden2tag(features_tmp)


class SequenceLabelForSO(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForSO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, hidden_size // 2)
        self.hidden2tag_sub = nn.Linear(hidden_size // 2, self.tag_size)
        self.hidden2tag_obj = nn.Linear(hidden_size // 2, self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output


# noinspection PyUnusedLocal
def get_auto_prgc_re_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model, base_model_name = MODEL_MAP[model_type]

    class PRGC(parent_model):
        """
        基于`BERT`的`PRGC`关系抽取模型
        + 📖 模型的整体思路将三元组抽取问题转化为潜在关系提取、主体-客体抽取和主体-客体对齐
        + 📖 潜在关系提取采用多标签分类预测句子中所有潜在的关系类型
        + 📖 对于每一个潜在关系，通过`BIO`标注预测所有候选的主体-客体对
        + 📖 通过表格填充学习主体-客体首首对齐的关系，筛选不合理的主体-客体对

        Args:
            `config`: 模型的配置对象

        Reference:
            ⭐️ [PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction.](https://aclanthology.org/2021.acl-long.486.pdf)
            🚀 [Official Code](https://github.com/hy-struggle/PRGC)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            setattr(self, base_model_name, base_model(config, add_pooling_layer=False))

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            if config.emb_fusion == 'concat':
                self.sequence_tagging_sub = MultiNonLinearClassifier(
                    config.hidden_size * 2,
                    3,
                    config.hidden_dropout_prob
                )
                self.sequence_tagging_obj = MultiNonLinearClassifier(
                    config.hidden_size * 2,
                    3,
                    config.hidden_dropout_prob
                )
            else:
                self.sequence_tagging_sum = SequenceLabelForSO(
                    config.hidden_size,
                    3,
                    config.hidden_dropout_prob
                )

            # 潜在关系判断
            self.rel_judgement = MultiNonLinearClassifier(
                config.hidden_size,
                config.num_predicates,
                config.hidden_dropout_prob
            )
            # 关系嵌入
            self.rel_embedding = nn.Embedding(config.num_predicates, config.hidden_size)

            if self.config.corres_mode == 'biaffine':
                self.U = torch.nn.Parameter(
                    torch.randn(
                        config.biaffine_hidden_size,
                        1,
                        config.biaffine_hidden_size
                    )
                )
                self.start_encoder = torch.nn.Sequential(
                    torch.nn.Linear(in_features=config.hidden_size,
                                    out_features=config.biaffine_hidden_size),
                    torch.nn.ReLU()
                )
                self.end_encoder = torch.nn.Sequential(
                    torch.nn.Linear(in_features=config.hidden_size,
                                    out_features=config.biaffine_hidden_size),
                    torch.nn.ReLU()
                )
            else:
                # global correspondence
                self.global_corres = MultiNonLinearClassifier(
                    config.hidden_size * 2,
                    1,
                    config.hidden_dropout_prob
                )

            # Initialize weights and apply final processing
            self.post_init()

        @staticmethod
        def masked_avgpool(sent, mask):
            """ 平均池化所有token的特征
            """
            mask_ = mask.masked_fill(mask == 0, -1e9).float()
            score = torch.softmax(mask_, -1)
            return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

        def get_corres(self, sequence_output):
            seq_len = sequence_output.shape[1]
            if self.config.corres_mode == 'biaffine':
                sub_extend = self.start_encoder(sequence_output)
                obj_extend = self.end_encoder(sequence_output)

                corres_pred = torch.einsum('bxi, ioj, byj->bxyo', sub_extend, self.U, obj_extend).squeeze(-1)
            else:
                sub_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, l, l, h)
                obj_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, l, l, h)
                # batch x seq_len x seq_len x 2*hidden
                corres_pred = torch.cat([sub_extend, obj_extend], 3)
                # bs x seq_len x seq_len
                corres_pred = self.global_corres(corres_pred).squeeze(-1)
            return corres_pred

        # noinspection PyUnboundLocalVariable
        def get_sub_obj(self, sequence_output, rel_emb):
            if self.config.emb_fusion == 'concat':
                decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
                output_sub = self.sequence_tagging_sub(decode_input)
                output_obj = self.sequence_tagging_obj(decode_input)

            elif self.config.emb_fusion == 'sum':
                decode_input = sequence_output + rel_emb
                output_sub, output_obj = self.sequence_tagging_sum(decode_input)
            return output_sub, output_obj

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            seq_labels: Optional[torch.Tensor] = None,
            rel_labels: Optional[torch.Tensor] = None,
            potential_rels: Optional[torch.Tensor] = None,
            corres_labels: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            offset_mapping: Optional[List[Any]] = None,
            target: Optional[List[Any]] = None,
        ) -> RelationExtractionOutput:

            outputs = getattr(self, base_model_name)(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            sequence_output = self.dropout(outputs[0])  # [batch_size, seq_len, hidden_size]

            _, seq_len, h = sequence_output.size()
            corres_pred = self.get_corres(sequence_output)

            # (bs, h)
            h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
            # (bs, rel_num)
            rel_pred = self.rel_judgement(h_k_avg)

            loss, predictions = None, None
            if self.training and potential_rels is not None:
                # (bs, h)
                rel_emb = self.rel_embedding(potential_rels)
                # relation embedding vector fusion
                rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)
                output_sub, output_obj = self.get_sub_obj(sequence_output, rel_emb)

                if seq_labels is not None and corres_labels is not None:
                    loss = self.compute_loss(
                        [output_sub, output_obj, corres_pred, rel_pred, attention_mask],
                        [seq_labels, corres_labels, rel_labels]
                    )

            else:
                predictions = self.decode(sequence_output, rel_pred, corres_pred, attention_mask, texts, offset_mapping)

            return RelationExtractionOutput(
                loss=loss,
                logits=None,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, sequence_output, rel_pred, corres_pred, attention_mask, texts, offset_mapping):
            bs, seq_len, _ = sequence_output.shape
            corres_mask = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)
            rel_threshold = getattr(self.config, "rel_threshold", 0.1)
            corres_threshold = getattr(self.config, "corres_threshold", 0.5)

            id2label = {int(v): k for k, v in self.config.label2id.items()}
            id2predicate = {int(v): k for k, v in self.config.predicate2id.items()}

            decode_labels = []
            for i in range(bs):
                spoes = set()
                rels = torch.where(rel_pred[i] > rel_threshold)[0]
                if len(rels) > 0:
                    seq_out = sequence_output[i].unsqueeze(0).expand(len(rels), -1, -1)
                    rel_emb = self.rel_embedding(rels)
                    rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, -1)
                    output_sub, output_obj = self.get_sub_obj(seq_out, rel_emb)

                    pred_seq_sub = torch.argmax(output_sub, dim=-1)
                    pred_seq_obj = torch.argmax(output_obj, dim=-1)

                    _corres_pred = torch.sigmoid(corres_pred[i]) * corres_mask[i]
                    pre_corres = torch.where(_corres_pred > corres_threshold, torch.ones_like(_corres_pred),
                                             torch.zeros_like(_corres_pred))

                    rels = rels.detach().cpu().numpy()
                    pred_seq_sub = tensor_to_numpy(pred_seq_sub)
                    pred_seq_obj = tensor_to_numpy(pred_seq_obj)
                    pre_corres = tensor_to_numpy(pre_corres)
                    spoes = self.get_triples(rels, pred_seq_sub, pred_seq_obj, pre_corres,
                                             texts[i], offset_mapping[i], id2label, id2predicate)
                decode_labels.append(spoes)

            return decode_labels

        def get_triples(self, rels, pred_seq_sub, pred_seq_obj, pre_corres, text, mapping, id2label, id2predicate):
            triples = set()
            for i in range(len(rels)):
                seq_sub_tags = [id2label[t] for t in pred_seq_sub[i]]
                seq_obj_tags = [id2label[t] for t in pred_seq_obj[i]]
                subjects, objects = get_entities(seq_sub_tags), get_entities(seq_obj_tags)
                for h in subjects:
                    for t in objects:
                        if pre_corres[h[1]][t[1]] == 1:
                            triples.add((
                                id2predicate[rels[i]],
                                text[mapping[h[1]][0]: mapping[h[2]][1]],
                                text[mapping[t[1]][0]: mapping[t[2]][1]]
                            ))
            return triples

        def compute_loss(self, inputs, targets):
            output_sub, output_obj, corres_pred, rel_pred, mask = inputs
            seq_labels, corres_labels, rel_labels = targets

            batch_size = mask.shape[0]
            corres_mask = mask.unsqueeze(-1) * mask.unsqueeze(1)

            mask = mask.view(-1)
            # sequence label loss
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss_seq_sub = (loss_fct(output_sub.view(-1, 3), seq_labels[:, 0, :].reshape(-1)) * mask).sum() / mask.sum()
            loss_seq_obj = (loss_fct(output_obj.view(-1, 3), seq_labels[:, 1, :].reshape(-1)) * mask).sum() / mask.sum()
            loss_seq = (loss_seq_sub + loss_seq_obj) / 2

            corres_pred = corres_pred.view(batch_size, -1)
            corres_mask = corres_mask.view(batch_size, -1)
            corres_tags = corres_labels.view(batch_size, -1)

            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss_matrix = (loss_fct(corres_pred, corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

            loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
            loss_rel = loss_fct(rel_pred, rel_labels.float())
            return loss_seq + loss_matrix + loss_rel

    return PRGC


def get_prgc_model_config(predicates, **kwargs):
    predicate2id = {v: i for i, v in enumerate(predicates)}

    tags = ["O", "B-ENT", "I-ENT"]
    label2id = {t: idx for idx, t in enumerate(tags)}

    model_config = {
        "num_predicates": len(predicates), "predicate2id": predicate2id,
        "label2id": label2id, "rel_threshold": 0.1, "corres_threshold": 0.5,
        "emb_fusion": "concat", "corres_mode": "biaffine", "biaffine_hidden_size": 128,
    }
    model_config.update(kwargs)
    return model_config

import torch
import torch.nn as nn
from typing import Optional, List, Any
from transformers import PreTrainedModel
from lightningblocks.utils.tensor import sequence_padding, tensor_to_cpu
from lightningblocks.layers.crf import CRF
from lightningblocks.task.utils import get_entities, SequenceLabelingOutput, MODEL_MAP
from lightningblocks.losses.focal_loss import FocalLoss
from lightningblocks.losses.label_smoothing import LabelSmoothingCE


def get_auto_crf_ner_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model = MODEL_MAP[model_type]

    class CRFForNer(parent_model):
        """
        基于`BERT`的`CRF`实体识别模型
        + 📖 `BERT`编码器提取`token`的语义特征
        + 📖 `CRF`层学习标签之间的约束关系

        Args:
            `config`: 模型的配置对象
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.bert = base_model(config, add_pooling_layer=False)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)
            
            self.use_lstm = getattr(config, 'use_lstm', False)
            mid_hidden_size = getattr(config, 'mid_hidden_size', config.hidden_size // 3)
            if self.use_lstm:
                self.mid_layer = nn.LSTM(
                    config.hidden_size,
                    mid_hidden_size // 2,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                    dropout=classifier_dropout
                )

            self.classifier = nn.Linear(mid_hidden_size if self.use_lstm else config.hidden_size, config.num_labels)
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)

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
            if self.use_lstm:
                sequence_output, _ = self.mid_layer(sequence_output)

            logits = self.classifier(sequence_output)

            loss, predictions = None, None
            if labels is not None:
                loss = self.compute_loss([logits, labels, attention_mask])

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

        def decode(self, logits, mask, texts, offset_mapping):
            decode_ids = self.crf.decode(logits, mask).squeeze(0)  # (batch_size, seq_length)
            decode_ids, mask = tensor_to_cpu(decode_ids), tensor_to_cpu(mask)
            id2label = {int(v): k for k, v in self.config.label2id.items()}

            decode_labels = []
            for text, ids, mask, mapping in zip(texts, decode_ids, mask, offset_mapping):
                decode_label = [id2label[id.item()] for id, m in zip(ids, mask) if m > 0][:-1]  # [CLS], [SEP]
                decode_label = get_entities(decode_label)
                decode_label = [(l[0], mapping[l[1]][0], mapping[l[2]][1], text[mapping[l[1]][0]: mapping[l[2]][1]]) for l in decode_label]
                decode_labels.append(set(decode_label))

            return decode_labels

        def compute_loss(self, inputs):
            logits, labels, mask = inputs[:3]
            return -1 * self.crf(emissions=logits, tags=labels, mask=mask)

    return CRFForNer


def get_auto_cascade_crf_ner_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model = MODEL_MAP[model_type]

    class CascadeCRFForNer(parent_model):
        """
        基于`BERT`的层级`CRF`实体识别模型
        + 📖 `BERT`编码器提取`token`的语义特征
        + 📖 第一阶段`CRF`层学习`BIO`标签之间的约束关系抽取所有实体
        + 📖 第二阶段采用一个线性层对实体进行分类
        
        Args:
            `config`: 模型的配置对象
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.bert = base_model(config, add_pooling_layer=False)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.dense1 = nn.Linear(config.hidden_size, 3)
            self.crf = CRF(num_tags=3, batch_first=True)
            self.dense2 = nn.Linear(config.hidden_size, config.num_labels)

            # Initialize weights and apply final processing
            self.post_init()

        def get_entity_logits(self, sequence_output, entity_ids):
            btz, entity_count, _ = entity_ids.shape
            entity_ids = entity_ids.reshape(btz, -1, 1).repeat(1, 1, self.config.hidden_size)
            entity_states = torch.gather(sequence_output, dim=1, index=entity_ids).reshape(btz, entity_count, -1,
                                                                                           self.config.hidden_size)
            entity_states = torch.mean(entity_states, dim=2)  # 取实体首尾 `hidden_states` 的均值
            return self.dense2(entity_states)  # [btz, 实体个数，实体类型数]

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            entity_ids: Optional[torch.Tensor] = None,
            entity_labels: Optional[torch.Tensor] = None,
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
            logits = self.dense1(sequence_output)

            loss, predictions = None, None
            if labels is not None and entity_ids is not None and entity_labels is not None:
                entity_logits = self.get_entity_logits(sequence_output, entity_ids)
                loss = self.compute_loss([logits, entity_logits, entity_labels, labels, attention_mask])

            if not self.training and return_decoded_labels:  # 训练时无需解码
                predictions = self.decode(sequence_output, logits, attention_mask, texts, offset_mapping)

            return SequenceLabelingOutput(
                loss=loss,
                logits=logits,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, sequence_output, logits, mask, texts, offset_mapping):
            decode_ids = self.crf.decode(logits, mask).squeeze(0)  # (batch_size, seq_length)
            BIO_MAP = getattr(self.config, 'BIO_MAP', {0: "O", 1: "B-ENT", 2: "I-ENT"})
            id2label = {int(v): k for k, v in self.config.label2id.items()}

            entity_ids = []
            for ids, mask in zip(decode_ids, mask):
                decode_label = [BIO_MAP[id.item()] for id, m in zip(ids, mask) if m > 0][:-1]  # [CLS], [SEP]
                decode_label = get_entities(decode_label)
                if len(decode_label) > 0:
                    entity_ids.append([[l[1], l[2]] for l in decode_label])
                else:
                    entity_ids.append([[0, 0]])

            entity_ids = torch.from_numpy(sequence_padding(entity_ids)).to(sequence_output.device)
            entity_logits = self.get_entity_logits(sequence_output, entity_ids)
            entity_preds = torch.argmax(entity_logits, dim=-1)  # [btz, 实体个数]

            decode_labels = []
            entity_ids, entity_preds = tensor_to_cpu(entity_ids, entity_preds)
            for i, (entities, text, mapping) in enumerate(zip(entity_ids, texts, offset_mapping)):
                tmp = set()
                for j, ent in enumerate(entities):
                    s, e, p = ent[0].item(), ent[1].item(), entity_preds[i][j].item()
                    if s * e * p != 0:
                        _start, _end = mapping[s][0], mapping[e][1]
                        tmp.add((
                            id2label[p], _start, _end, text[_start: _end]
                        ))
                decode_labels.append(tmp)

            return decode_labels

        def compute_loss(self, inputs):
            logits, entity_logits, entity_labels, labels, mask = inputs[:5]
            loss = -1 * self.crf(emissions=logits, tags=entity_labels, mask=mask)
            loss += 4 * nn.CrossEntropyLoss(ignore_index=0)(entity_logits.view(-1, self.config.num_labels),
                                                            labels.flatten())
            return loss

    return CascadeCRFForNer


def get_auto_softmax_ner_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:
    base_model, parent_model = MODEL_MAP[model_type]

    class SoftmaxForNer(parent_model):
        """
        基于`BERT`的`Softmax`实体识别模型

        Args:
            `config`: 模型的配置对象
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.bert = base_model(config, add_pooling_layer=False)

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            loss_type = getattr(config, 'loss_type', 'cross_entropy')

            loss_fcts = {
                'cross_entropy': nn.CrossEntropyLoss(ignore_index=0),
                'focal_loss': FocalLoss(config.num_labels),
                'label_smoothing_ce': LabelSmoothingCE(ignore_index=0)
            }
            self.loss_fct = loss_fcts.get(loss_type, 'cross_entropy')

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
            logits = self.classifier(sequence_output)

            loss, predictions = None, None
            if labels is not None:
                loss = self.compute_loss([logits, labels, attention_mask])

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

        def decode(self, logits, mask, texts, offset_mapping):
            logits, mask = tensor_to_cpu(logits), tensor_to_cpu(mask)
            decode_ids = torch.argmax(logits, -1)  # (batch_size, seq_length)
            id2label = {int(v): k for k, v in self.config.label2id.items()}

            decode_labels = []
            for text, ids, mask, mapping in zip(texts, decode_ids, mask, offset_mapping):
                decode_label = [id2label[id.item()] for id, m in zip(ids, mask) if m > 0][:-1]  # [CLS], [SEP]
                decode_label = get_entities(decode_label)
                decode_label = [(l[0], mapping[l[1]][0], mapping[l[2]][1], text[mapping[l[1]][0]: mapping[l[2]][1]]) for
                                l in decode_label]
                decode_labels.append(set(decode_label))
            return decode_labels

        def compute_loss(self, inputs):
            logits, labels, attention_mask = inputs[:3]
            active_loss = attention_mask.view(-1) == 1

            active_logits = logits.view(-1, self.config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            return self.loss_fct(active_logits, active_labels)

    return SoftmaxForNer


def get_softmax_model_config(labels, **kwargs):
    """Softmax实体识别模型的主要参数配置
    """
    bio_labels = ['O'] + [f"B-{l}" for l in labels] + [f"I-{l}" for l in labels]
    label2id = {v: i for i, v in enumerate(bio_labels)}
    model_config = {
        "num_labels": len(bio_labels), "label2id": label2id,
    }
    model_config.update(kwargs)
    return model_config


def get_crf_model_config(labels, **kwargs):
    """CRF实体识别模型的主要参数配置
    """
    model_config = get_softmax_model_config(labels, **kwargs)
    special_config = {"use_lstm": False, "mid_hidden_size": 256}
    model_config.update(special_config)
    return model_config


def get_cascade_crf_ner_model_config(labels, **kwargs):
    """Cascade CRF实体识别模型的主要参数配置
    """
    labels = ['O'] + labels
    label2id = {v: i for i, v in enumerate(labels)}
    model_config = {
        "num_labels": len(labels), "label2id": label2id,
    }
    model_config.update(kwargs)
    return model_config

from typing import Optional, List, Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import PreTrainedModel

from ...utils import SequenceLabelingOutput, MODEL_MAP
from ....layers.conv import DilateConvLayer
from ....layers.layer_norm import LayerNorm
from ....layers.mlp import MLP
from ....utils.tensor import tensor_to_cpu


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        
        self.bias_x = bias_x
        self.bias_y = bias_y
        
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi, oij, byj->boxy', x, self.weight, y)
        return s.permute(0, 2, 3, 1)


class CoPredictor(nn.Module):
    def __init__(self, num_labels, hidden_size, biaffine_size, channels, ffn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(hidden_size, biaffine_size, dropout)
        self.mlp2 = MLP(hidden_size, biaffine_size, dropout)

        self.biaffine = Biaffine(biaffine_size, num_labels)
        self.mlp_rel = MLP(channels, ffn_hid_size, dropout)
        self.linear = nn.Linear(ffn_hid_size, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        ent_sub = self.dropout(self.mlp1(x))
        ent_obj = self.dropout(self.mlp2(y))
        o1 = self.biaffine(ent_sub, ent_obj)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2


# noinspection PyUnusedLocal
def get_auto_w2ner_ner_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model, base_model_name = MODEL_MAP[model_type]

    class W2Ner(parent_model):
        """
        基于`BERT`的`W2Ner`实体识别模型
        + 📖 将实体抽取任务统一起来，可以解决嵌套实体和不连续实体的抽取
        + 📖 将单词对关系进行建模，使用卷积、距离嵌入等抽取表格特征
        
        Args:
            `config`: 模型的配置对象
        
        Reference:
            ⭐️ [Unified Named Entity Recognition as Word-Word Relation Classification.](https://arxiv.org/pdf/2112.10070.pdf)
            🚀 [Official Code](https://github.com/ljynlp/W2NER)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            setattr(self, base_model_name, base_model(config, add_pooling_layer=False))

            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)

            dist_emb_size, type_emb_size = getattr(config, "dist_emb_size", 20), getattr(config, "type_emb_size", 20)
            self.dis_embs = nn.Embedding(20, dist_emb_size)
            self.reg_embs = nn.Embedding(3, type_emb_size)

            lstm_hidden_size = getattr(config, "lstm_hidden_size", 512)
            self.lstm = nn.LSTM(config.hidden_size, lstm_hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)

            conv_input_size = lstm_hidden_size + dist_emb_size + type_emb_size
            conv_hidden_size = getattr(config, "conv_hidden_size", 96)
            self.conv = DilateConvLayer(conv_input_size, conv_hidden_size, dilation=[1, 2, 3], dropout=0.5)
            
            biaffine_size, ffn_hidden_size = getattr(config, "biaffine_size", 512), getattr(config, "ffn_hidden_size", 288)
            self.predictor = CoPredictor(config.num_labels, lstm_hidden_size, biaffine_size, conv_hidden_size * 3, ffn_hidden_size, 0.33)

            self.cln = LayerNorm(lstm_hidden_size, conditional_size=lstm_hidden_size)

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            pieces2word: Optional[torch.Tensor] = None,
            dist_inputs: Optional[torch.Tensor] = None,
            input_lengths: Optional[torch.Tensor] = None,
            grid_mask: Optional[torch.Tensor] = None,
            grid_labels: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            target: Optional[List[Any]] = None,
            return_decoded_labels: Optional[bool] = True,
        ) -> SequenceLabelingOutput:

            outputs = getattr(self, base_model_name)(
                input_ids,
                output_attentions=output_attentions,
                output_hidden_states=self.config.use_last_4_layers,
            )

            if self.config.use_last_4_layers:
                sequence_output = torch.stack(outputs[2][-4:], dim=-1).mean(-1)
            else:
                sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)  # [batch_size, seq_len, hidden_size]

            length = pieces2word.size(1)
            min_value = torch.min(sequence_output).item()

            # Max pooling word representations from pieces
            sequence_output = sequence_output.unsqueeze(1).expand(-1, length, -1, -1)
            sequence_output = torch.masked_fill(sequence_output, pieces2word.eq(0).unsqueeze(-1), min_value)
            word_reps, _ = torch.max(sequence_output, dim=2)

            word_reps = self.dropout(word_reps)
            packed_embs = pack_padded_sequence(word_reps, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_outs, (hidden, _) = self.lstm(packed_embs)
            word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=input_lengths.max())

            cln = self.cln([word_reps.unsqueeze(2), word_reps])
            
            dis_emb = self.dis_embs(dist_inputs)
            tril_mask = torch.tril(grid_mask.clone().long())
            reg_inputs = tril_mask + grid_mask.clone().long()
            reg_emb = self.reg_embs(reg_inputs)

            conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
            conv_inputs = torch.masked_fill(conv_inputs, grid_mask.eq(0).unsqueeze(-1), 0.0)
            conv_outputs = self.conv(conv_inputs)
            conv_outputs = torch.masked_fill(conv_outputs, grid_mask.eq(0).unsqueeze(-1), 0.0)
            
            logits = self.predictor(word_reps, word_reps, conv_outputs)

            loss, predictions = None, None
            if grid_labels is not None:
                loss = self.compute_loss([logits, grid_labels, grid_mask])

            if not self.training and return_decoded_labels:
                predictions = self.decode(logits, input_lengths, texts)

            return SequenceLabelingOutput(
                loss=loss,
                logits=logits,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def decode(self, logits, input_lengths, texts):
            decode_labels = []
            logits, input_lengths = tensor_to_cpu(logits.argmax(-1)), tensor_to_cpu(input_lengths)
            id2label = {int(v): k for k, v in self.config.label2id.items()}

            for _logits, length, text in zip(logits, input_lengths, texts):
                forward_dict, head_dict, ht_type_dict = {}, {}, {}
                for i in range(length):
                    for j in range(i + 1, length):
                        if _logits[i, j].item() == 1:  # NNW
                            if i not in forward_dict:
                                forward_dict[i] = [j]
                            else:
                                forward_dict[i].append(j)
                                
                for i in range(length):
                    for j in range(i, length):
                        if _logits[j, i].item() > 1:  # THW
                            ht_type_dict[(i, j)] = _logits[j, i].item()
                            if i not in head_dict:
                                head_dict[i] = {j}
                            else:
                                head_dict[i].add(j)

                predicts = []

                def find_entity(key, entity, tails):
                    entity.append(key)
                    if key in tails:
                        predicts.append(entity.copy())
                    if key not in forward_dict:
                        entity.pop()
                        return
                    for k in forward_dict[key]:
                        find_entity(k, entity, tails)
                    entity.pop()

                for head in head_dict:
                    find_entity(head, [], head_dict[head])

                entities = set()
                for _entity in predicts:
                    entities.add((
                        id2label[ht_type_dict[(_entity[0], _entity[-1])]],
                        _entity[0],
                        _entity[-1] + 1,
                        ''.join([text[i] for i in _entity]),
                    ))
                decode_labels.append(entities)
            
            return decode_labels

        def compute_loss(self, inputs):
            logits, labels, mask = inputs[:3]
            active_loss = mask.view(-1) == 1
            active_logits = logits.reshape(-1, self.config.num_labels)

            loss_fct = nn.CrossEntropyLoss()
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            return loss_fct(active_logits, active_labels.long())

    return W2Ner


def get_w2ner_model_config(labels, **kwargs):
    w2ner_labels = ["NONE", "NNW"] + labels
    label2id = {v: i for i, v in enumerate(w2ner_labels)}
    model_config = {
        "num_labels": len(w2ner_labels), "label2id": label2id,
        "use_last_4_layers": False, "dist_emb_size": 20, "type_emb_size": 20,
        "lstm_hidden_size": 512, "conv_hidden_size": 96, "biaffine_size": 512, "ffn_hidden_size": 288,
    }
    model_config.update(kwargs)
    return model_config

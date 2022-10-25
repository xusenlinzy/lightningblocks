import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from typing import Optional, List, Any
from transformers import PreTrainedModel
from lightningnlp.task.utils.decode_utils import filter_clashed_by_priority
from lightningnlp.utils.tensor import tensor_to_cpu, seq_len_to_mask, tensor_to_list
from lightningnlp.task.utils import SequenceLabelingOutput, MODEL_MAP


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim = dim
        self.eps = 1e-6

    def forward(self, x):
        mu = x.mean(dim=self.dim, keepdim=True)
        s = (x - mu).pow(2).mean(dim=self.dim, keepdim=True)
        x = (x - mu) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1):
        super(MaskedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False,
                                groups=groups)

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        return self.conv2d(x)


class MaskedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, depth=3):
        super(MaskedCNN, self).__init__()

        layers = []
        for _ in range(depth):
            layers.extend([
                MaskedConv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2),
                LayerNorm((1, in_channels, 1, 1), dim=1),
                nn.GELU(),
            ])
        layers.append(MaskedConv2d(in_channels, out_channels, kernel_size=3, padding=3//2))
        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        _x = x
        for layer in self.cnns:
            if isinstance(layer, LayerNorm):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x, mask)
            else:
                x = layer(x)
        return _x


class MultiHeadBiaffine(nn.Module):
    def __init__(self, hidden_size, out_size=None, num_heads=4):
        super(MultiHeadBiaffine, self).__init__()
        assert hidden_size % num_heads == 0
        in_head_dim = hidden_size // num_heads

        out_size = hidden_size if out_size is None else out_size
        assert out_size % num_heads == 0
        out_head_dim = out_size // num_heads

        self.num_heads = num_heads
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.num_heads, out_head_dim, in_head_dim, in_head_dim)))
        self.out_size = out_size

    def forward(self, h, v):
        bsz, max_len, dim = h.size()
        h = h.reshape(bsz, max_len, self.num_heads, -1)
        v = v.reshape(bsz, max_len, self.num_heads, -1)
        w = torch.einsum('blhx, hdxy, bkhy -> bhdlk', h, self.W, v)
        return w.reshape(bsz, self.out_size, max_len, max_len)


def get_auto_cnn_ner_model(
    model_type: str = "bert",
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> PreTrainedModel:

    base_model, parent_model = MODEL_MAP[model_type]

    class CNNForNer(parent_model):
        """
        基于`CNN`的实体识别模型

        Args:
            `config`: 模型的配置对象

        Reference:
            ⭐️ [An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition.](https://arxiv.org/abs/2208.04534)
            🚀 [Code](https://github.com/yhcc/CNN_Nested_NER)
        """

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.bert = base_model(config, add_pooling_layer=False)
            self.dropout = nn.Dropout(0.4)

            size_embed_dim = getattr(config, 'size_embed_dim', 0)
            biaffine_size = getattr(config, 'biaffine_size', 200)
            if size_embed_dim != 0:
                n_pos = 30
                self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
                _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
                _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
                _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
                self.register_buffer("span_size_ids", _span_size_ids.long())
                hsz = biaffine_size * 2 + size_embed_dim + 2
            else:
                hsz = biaffine_size * 2 + 2

            self.head_mlp = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(config.hidden_size, biaffine_size),
                nn.LeakyReLU(),
            )
            self.tail_mlp = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(config.hidden_size, biaffine_size),
                nn.LeakyReLU(),
            )

            num_heads = getattr(config, 'num_heads', 4)
            cnn_hidden_size = getattr(config, 'cnn_hidden_size', 200)
            if num_heads > 0:
                self.multi_head_biaffine = MultiHeadBiaffine(biaffine_size, cnn_hidden_size, num_heads)
            else:
                self.U = nn.Parameter(torch.randn(cnn_hidden_size, biaffine_size, biaffine_size))
                torch.nn.init.xavier_normal_(self.U.data)

            self.W = torch.nn.Parameter(torch.empty(cnn_hidden_size,  hsz))
            torch.nn.init.xavier_normal_(self.W.data)

            kernel_size, cnn_depth = getattr(config, "kernel_size", 3), getattr(config, "cnn_depth", 3)
            if cnn_depth > 0:
                self.cnn = MaskedCNN(cnn_hidden_size, cnn_hidden_size, kernel_size=kernel_size, depth=cnn_depth)

            self.fc = nn.Linear(cnn_hidden_size, config.num_labels)

            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            indexes: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            texts: Optional[List[str]] = None,
            target: Optional[List[Any]] = None,
            return_decoded_labels: Optional[bool] = True,
        ) -> SequenceLabelingOutput:

            outputs = self.bert(
                input_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            state = scatter_max(outputs[0], index=indexes, dim=1)[0][:, 1:]  # bsz x word_len x hidden_size
            lengths, _ = indexes.max(dim=-1)

            head_state = self.head_mlp(state)
            tail_state = self.tail_mlp(state)

            if hasattr(self, "U"):
                scores1 = torch.einsum("bxi, oij, byj -> boxy", head_state, self.U, tail_state)
            else:
                scores1 = self.multi_head_biaffine(head_state, tail_state)

            head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
            tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)

            affined_cat = torch.cat(
                [self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)],
                dim=-1,
            )

            if hasattr(self, "size_embedding"):
                size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
                affined_cat = torch.cat(
                    [affined_cat, self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)], dim=-1)

            scores2 = torch.einsum("bmnh, kh -> bkmn", affined_cat, self.W)  # bsz x dim x L x L
            scores = scores2 + scores1  # bsz x dim x L x L

            if hasattr(self, "cnn"):
                mask = seq_len_to_mask(lengths)
                mask = mask[:, None] * mask.unsqueeze(-1)
                pad_mask = mask[:, None].eq(0)
                u_scores = scores.masked_fill(pad_mask, 0)
                u_scores = self.cnn(u_scores, pad_mask)  # bsz, num_labels, max_len, max_len = u_scores.size()
                scores = u_scores + scores

            scores = self.fc(scores.permute(0, 2, 3, 1))

            loss, predictions = None, None
            if labels is not None:
                scores = scores.reshape(-1)
                labels = labels.reshape(-1)

                mask = labels.ne(-100).float().view(input_ids.size(0), -1)
                loss = F.binary_cross_entropy_with_logits(scores, labels.float(), reduction='none')
                loss = ((loss.view(input_ids.size(0), -1) * mask).sum(dim=-1)).mean()

            if not self.training and return_decoded_labels:  # 训练时无需解码
                predictions = self.decode(scores, lengths, texts)

            return SequenceLabelingOutput(
                loss=loss,
                logits=scores,
                predictions=predictions,
                groundtruths=target,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        @staticmethod
        def spans_from_upper_triangular(seq_len: int):
            """Spans from the upper triangular area.
            """
            for start in range(seq_len):
                for end in range(start, seq_len):
                    yield start, end

        def _decode(self, scores, lengths, allow_nested=False, thresh=0.5):
            batch_chunks = []
            for idx, (_scores, l) in enumerate(zip(scores, lengths)):
                curr_non_mask = scores.new_ones(l, l, dtype=bool).triu()
                tmp_scores = _scores[:l, :l][curr_non_mask]

                confidences, label_ids = tmp_scores, tmp_scores >= thresh
                labels = [i for i in label_ids]
                chunks = [(label, start, end) for label, (start, end) in
                          zip(labels, self.spans_from_upper_triangular(l)) if label != 0]
                confidences = [conf for label, conf in zip(labels, confidences) if label != 0]

                assert len(confidences) == len(chunks)
                chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
                chunks = filter_clashed_by_priority(chunks, allow_nested=allow_nested)
                if len(chunks):
                    batch_chunks.append(set([(s, e, l) for l, s, e in chunks]))
                else:
                    batch_chunks.append(set())
            return batch_chunks

        def decode(self, scores, lengths, texts):
            all_entity_list = []
            scores, lengths = tensor_to_cpu(torch.sigmoid(scores)), tensor_to_list(lengths)
            scores = (scores + scores.transpose(1, 2)) / 2
            span_pred = scores.max(dim=-1)[0]

            decode_thresh = getattr(self.config, "decode_thresh", 0.5)
            allow_nested = getattr(self.config, "allow_nested", True)
            id2label = {int(v): k for k, v in self.config.label2id.items()}

            span_ents = self._decode(span_pred, lengths, allow_nested=allow_nested, thresh=decode_thresh)
            for span_ent, _scores, text in zip(span_ents, scores, texts):
                entity_set = set()
                for s, e, l in span_ent:
                    score = _scores[s, e]
                    _type = score.argmax()
                    if score[_type] >= decode_thresh:
                        entity_set.add((id2label[_type.item()], s, e + 1, text[s: e + 1]))
                all_entity_list.append(entity_set)

            return all_entity_list

    return CNNForNer


def get_cnn_model_config(labels, **kwargs):
    label2id = {v: i for i, v in enumerate(labels)}
    model_config = {
        "num_labels": len(labels), "label2id": label2id, "size_embed_dim": 25,
        "biaffine_size": 200, "num_heads": 4, "cnn_hidden_size": 200,
        "kernel_size": 3, "decode_thresh": 0.5, "cnn_depth": 3,
    }
    model_config.update(kwargs)
    return model_config

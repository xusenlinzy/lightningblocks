import math
import torch
import torch.nn as nn
from .position import *


class CosAttention(nn.Module):
    def __init__(self):
        super(CosAttention, self).__init__()

    def forward(self, q, k, v):
        """
        q: (batchsize, hidden_dim)
        k: (batchsize, seqlen, hidden_dim)
        v: (batchsize, seqlen, hidden_dim)
        """
        seq_len = k.size()[1]
        q_output = q.unsqueeze(1).repeat(1, seq_len, 1)
        cos_sim = torch.cosine_similarity(q_output, k, -1)
        cos_sim = cos_sim.unsqueeze(-1)
        return v * cos_sim


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 attention_scale=True,
                 return_attention_scores=False,
                 **kwargs):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hidden_size % num_attention_heads == 0

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.a_bias, self.p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')

        if self.p_bias == 'typical_relative':  # nezha
            self.relative_positions_encoding = RelativePositionsEncoding(qlen=kwargs.get('max_position'),
                                                                         klen=kwargs.get('max_position'),
                                                                         embedding_size=self.attention_head_size,
                                                                         max_relative_position=kwargs.get(
                                                                             'max_relative_position'))
        elif self.p_bias == 'rotary':  # roformer
            self.relative_positions_encoding = RoPEPositionEncoding(max_position=kwargs.get('max_position'),
                                                                    embedding_size=self.attention_head_size)
        elif self.p_bias == 't5_relative':  # t5
            self.relative_positions = T5RelativePositionsEncoding(qlen=kwargs.get('max_position'),
                                                                  klen=kwargs.get('max_position'),
                                                                  relative_attention_num_buckets=kwargs.get(
                                                                      'relative_attention_num_buckets'),
                                                                  is_decoder=kwargs.get('is_decoder'))
            self.relative_positions_encoding = nn.Embedding(kwargs.get('relative_attention_num_buckets'),
                                                            self.num_attention_heads)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        # hidden_states shape: [batch_size, seq_q, hidden_size]
        # attention_mask shape: [batch_size, 1, 1, seq_q] 或者 [batch_size, 1, seq_q, seq_q]
        # encoder_hidden_states shape: [batch_size, seq_k, hidden_size]
        # encoder_attention_mask shape: [batch_size, 1, 1, seq_k]

        mixed_query_layer = self.q(hidden_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.k(encoder_hidden_states)
            mixed_value_layer = self.v(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.k(hidden_states)
            mixed_value_layer = self.v(hidden_states)
        # mixed_query_layer shape: [batch_size, query_len, hidden_size]
        # mixed_query_layer shape: [batch_size, key_len, hidden_size]
        # mixed_query_layer shape: [batch_size, value_len, hidden_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # query_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        # key_layer shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        # value_layer shape: [batch_size, num_attention_heads, value_len, attention_head_size]

        if self.p_bias == 'rotary':
            query_layer = self.relative_positions_encoding(query_layer)
            key_layer = self.relative_positions_encoding(key_layer)

        # 交换k的最后两个维度，然后q和k执行点积, 获得attention score
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # attention_scores shape: [batch_size, num_attention_heads, query_len, key_len]
        if (self.p_bias == 'typical_relative') and hasattr(self, 'relative_positions_encoding'):
            relations_keys = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[
                -1])  # [to_seq_len, to_seq_len, d_hid]
            # 旧实现，方便读者理解维度转换
            # query_layer_t = query_layer.permute(2, 0, 1, 3)
            # query_layer_r = query_layer_t.contiguous().view(from_seq_length, batch_size * num_attention_heads, self.attention_head_size)
            # key_position_scores = torch.matmul(query_layer_r, relations_keys.permute(0, 2, 1))
            # key_position_scores_r = key_position_scores.view(from_seq_length, batch_size, num_attention_heads, from_seq_length)
            # key_position_scores_r_t = key_position_scores_r.permute(1, 2, 0, 3)
            # 新实现
            key_position_scores_r_t = torch.einsum('bnih,ijh->bnij', query_layer, relations_keys)
            attention_scores = attention_scores + key_position_scores_r_t
        elif (self.p_bias == 't5_relative') and hasattr(self, 'relative_positions_encoding'):
            relations_keys = self.relative_positions(attention_scores.shape[-1], attention_scores.shape[-1])
            key_position_scores_r_t = self.relative_positions_encoding(relations_keys).permute([2, 0, 1]).unsqueeze(0)
            attention_scores = attention_scores + key_position_scores_r_t

        # 是否进行attention scale
        if self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 执行attention mask，对于mask为0部分的attention mask，
        # 值为-1e10，经过softmax后，attention_probs几乎为0，所以不会attention到mask为0的部分
        if attention_mask is not None:
            # attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)
            attention_mask = (1.0 - attention_mask) * -10000.0  # 所以传入的mask的非padding部分为1, padding部分为0
            attention_scores = attention_scores + attention_mask

        # 将attention score 归一化到0-1
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # [batch_size, num_attention_heads, query_len, attention_head_size]

        if (self.p_bias == 'typical_relative') and hasattr(self, 'relative_positions_encoding'):
            relations_values = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[-1])
            # 旧实现，方便读者理解维度转换
            # attention_probs_t = attention_probs.permute(2, 0, 1, 3)
            # attentions_probs_r = attention_probs_t.contiguous().view(from_seq_length, batch_size * num_attention_heads, to_seq_length)
            # value_position_scores = torch.matmul(attentions_probs_r, relations_values)
            # value_position_scores_r = value_position_scores.view(from_seq_length, batch_size, num_attention_heads, self.attention_head_size)
            # value_position_scores_r_t = value_position_scores_r.permute(1, 2, 0, 3)
            # 新实现
            value_position_scores_r_t = torch.einsum('bnij,ijh->bnih', attention_probs, relations_values)
            context_layer = context_layer + value_position_scores_r_t

        # context_layer shape: [batch_size, query_len, num_attention_heads, attention_head_size]
        # transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，
        # 所以在调用view之前，需要contiguous来返回一个contiguous copy；
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 是否返回attention scores
        if self.return_attention_scores:
            # 这里返回的attention_scores没有经过softmax, 可在外部进行归一化操作
            return self.o(context_layer), attention_scores
        else:
            return self.o(context_layer)

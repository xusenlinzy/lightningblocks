import torch.nn as nn
from .layer_norm import LayerNorm
from .linears import PositionWiseFeedForward
from .attentions import MultiHeadAttentionLayer
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput


class BertLayer(nn.Module):
    """
    Transformer层:
    顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm
    注意: 1、以上都不计dropout层，并不代表没有dropout，每一层的dropout使用略有不同，注意区分
          2、原始的Transformer的encoder中的Feed Forward层一共有两层linear，
          config.intermediate_size的大小不仅是第一层linear的输出尺寸，也是第二层linear的输入尺寸
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 dropout_rate,
                 attention_probs_dropout_prob,
                 intermediate_size,
                 hidden_act,
                 is_dropout=False,
                 conditional_size=False,
                 **kwargs):
        super(BertLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads,
                                                          attention_probs_dropout_prob, **kwargs)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layerNorm1 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size)

        self.feedForward = PositionWiseFeedForward(hidden_size, intermediate_size, dropout_rate, hidden_act,
                                                   is_dropout=is_dropout)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layerNorm2 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size)
        self.is_decoder = kwargs.get('is_decoder')

        if self.is_decoder:
            self.crossAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads,
                                                          attention_probs_dropout_prob, **kwargs)
            self.dropout3 = nn.Dropout(dropout_rate)
            self.layerNorm3 = LayerNorm(hidden_size, eps=1e-12, conditional_size=conditional_size)

    def forward(self,
                hidden_states,
                attention_mask,
                conditional_emb=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        self_attn_output = self.multiHeadAttention(hidden_states,
                                                   attention_mask)  # self.decoder为true时候，这里的attention_mask是三角的
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        hidden_states = self.layerNorm1((hidden_states, conditional_emb))

        # cross attention
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attn_output = self.crossAttention(hidden_states, None, encoder_hidden_states, encoder_attention_mask)
            hidden_states = hidden_states + self.dropout3(cross_attn_output)
            hidden_states = self.layerNorm3((hidden_states, conditional_emb))

        self_attn_output2 = self.feedForward(hidden_states)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        hidden_states = self.layerNorm2((hidden_states, conditional_emb))
        return hidden_states


class T5Layer(BertLayer):
    """T5的Encoder的主体是基于Self-Attention的模块
    顺序：LN --> Att --> Add --> LN --> FFN --> Add
    """

    def __init__(self, *args, version='t5.1.0', **kwargs):
        super().__init__(*args, **kwargs)
        # 定义RMSnorm层
        self.layerNorm1 = LayerNorm(hidden_size=args[0], eps=1e-12, bias=False, mode='rmsnorm', **kwargs)
        self.layerNorm2 = LayerNorm(hidden_size=args[0], eps=1e-12, bias=False, mode='rmsnorm', **kwargs)

        # 删除对应的bias项
        self.multiHeadAttention.q.register_parameter('bias', None)
        self.multiHeadAttention.k.register_parameter('bias', None)
        self.multiHeadAttention.v.register_parameter('bias', None)
        self.multiHeadAttention.o.register_parameter('bias', None)

        # 如果是t5.1.1结构，则FFN层需要变更
        if version.endswith('t5.1.0'):
            self.feedForward.outputDense.register_parameter('bias', None)
            self.feedForward.intermediateDense.register_parameter('bias', None)
        elif version.endswith('t5.1.1'):
            kwargs['dropout_rate'] = args[2]
            kwargs['hidden_act'] = args[5]
            self.feedForward = self.T5PositionWiseFeedForward(hidden_size=args[0], intermediate_size=args[4], **kwargs)
        else:
            raise ValueError('T5 model only support t5.1.0 and t5.1.1')

        # decoder中间有crossAttention
        if self.is_decoder:
            self.layerNorm3 = LayerNorm(hidden_size=args[0], eps=1e-12, bias=False, mode='rmsnorm', **kwargs)
            self.crossAttention.q.register_parameter('bias', None)
            self.crossAttention.k.register_parameter('bias', None)
            self.crossAttention.v.register_parameter('bias', None)
            self.crossAttention.o.register_parameter('bias', None)
            if hasattr(self.crossAttention, 'relative_positions_encoding'):
                del self.crossAttention.relative_positions_encoding
                del self.crossAttention.relative_positions

    def forward(self,
                hidden_states,
                attention_mask,
                conditional_emb=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        # bert的layernorm是在attn/ffc之后，Openai-gpt2是在之前
        x = self.layerNorm1((hidden_states, conditional_emb))
        self_attn_output = self.multiHeadAttention(x, attention_mask)
        hidden_states = hidden_states + self.dropout1(self_attn_output)

        # cross attention
        if self.is_decoder and encoder_hidden_states is not None:
            x = self.layerNorm3((hidden_states, conditional_emb))
            cross_attn_output = self.crossAttention(x, None, encoder_hidden_states, encoder_attention_mask)
            hidden_states = hidden_states + self.dropout3(cross_attn_output)

        x = self.layerNorm2((hidden_states, conditional_emb))
        ffn_output = self.feedForward(x)
        hidden_states = hidden_states + self.dropout2(ffn_output)
        return hidden_states


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *args):
        return args[0]


#####################################################################################

def get_extended_attention_mask(attention_mask):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]

    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class TransformerEncoderLayer(nn.Module):
    """
    Args:
        config.hidden_size: 隐藏层维度
        config.num_attention_heads: 注意力head数
        config.attention_probs_dropout_prob: dropout
        position_embedding_type=None: 位置编码类型
        config.is_decoder: 是否为解码器

        config.intermediate_size: 中间层维度
        config.hidden_act: 中间层激活函数
        config.hidden_dropout_prob: 中间层dropout
    """

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask)
        attention_outputs = self.attention(hidden_states, attention_mask=attention_mask)
        attention_output = attention_outputs[0]
        outputs = attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_outputs = self.intermediate(attention_output)
        layer_output = self.output(intermediate_outputs, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask=None):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if encoder_attention_mask is not None:
            encoder_attention_mask = get_extended_attention_mask(encoder_attention_mask)

        cross_attention_outputs = self.crossattention(hidden_states=attention_output,
                                                      encoder_hidden_states=encoder_hidden_states,
                                                      encoder_attention_mask=encoder_attention_mask)

        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class ResidualDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask=None):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if encoder_attention_mask is not None:
            encoder_attention_mask = get_extended_attention_mask(encoder_attention_mask)

        cross_attention_outputs = self.crossattention(hidden_states=attention_output,
                                                      encoder_hidden_states=encoder_hidden_states,
                                                      encoder_attention_mask=encoder_attention_mask)

        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        layer_output = self.layernorm(layer_output + hidden_states)
        outputs = (layer_output,) + outputs
        return outputs

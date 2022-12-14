import math

import torch
import torch.nn as nn

from .layer_norm import LayerNorm
from .position import (
    SinusoidalPositionEncoding,
    RoPEPositionEncoding,
    RelativePositionsEncoding
)


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个`(start, end)`作为整体来进行判断
    详细介绍：[GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)
    + 📖 `GlobalPointer`事实上就是`Multi-Head Attention`的一个简化版
    + 📖 有多少种实体就对应多少个`head`，相比`Multi-Head Attention`去掉了`V`相关的运算
    + 📖 没有相对位置信息，效果很差，所以加上了旋转式位置编码`RoPE`
    + 📖 为了解决类别不平衡问题，使用推广的多标签分类的交叉熵损失函数
    
    Args:
        `heads`: 实体类型数目
        `head_size`: 计算实体`(start, end)`得分时`query`和`key`的向量维度
        `hidden_size``: 序列隐藏层状态的维度
        `use_rope`: 是否使用`rope`相对位置编码
    """

    def __init__(self, hidden_size, head_size, heads, max_len=512, use_rope=True, tri_mask=True):
        super(GlobalPointer, self).__init__()
        self.heads = heads  # 实体类别数目
        self.head_size = head_size  # q, k 向量的维度
        self.use_rope = use_rope  # 是否使用旋转式相对位置编码
        self.tri_mask = tri_mask
        self.linear = nn.Linear(hidden_size, self.head_size * self.heads * 2)

        if self.use_rope:
            self.position_encoding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, hidden_state, mask=None):
        """ b: batch size
            c: heads
            l: sequence length
            h: hiddden size
            d: head_size
        """
        # hidden_states:  (b, l, h)
        sequence_output = self.linear(hidden_state)  # (b, l, c * d * 2)
        sequence_output = torch.stack(
            torch.chunk(sequence_output, self.heads, dim=-1), dim=-2
        )  # [..., heads, head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., heads, head_size]

        # RoPE编码 (乘性编码) 
        if self.use_rope:
            qw = self.position_encoding(qw)
            kw = self.position_encoding(kw)

        # 计算内积
        logits = torch.einsum('bmhd, bnhd->bhmn', qw, kw)  # [btz, heads, seq_len, seq_len] 

        # 排除padding: method2
        if mask is not None:
            mask1 = 1 - mask[:, None, :, None]  # [btz, 1, seq_len, 1]
            mask2 = 1 - mask[:, None, None, :]  # [btz, 1, 1, seq_len]

            logits = logits.masked_fill(mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tri_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        # scale返回
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个`(start, end)`作为整体来进行判断
    详细介绍：[Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877)
    """

    def __init__(self, hidden_size, head_size, heads, max_len=512, use_rope=True, tri_mask=True):
        super(EfficientGlobalPointer, self).__init__()
        self.heads = heads  # 实体类别数目
        self.head_size = head_size  # q, k 向量的维度
        self.use_rope = use_rope  # 是否使用旋转式相对位置编码
        self.tri_mask = tri_mask

        self.linear1 = nn.Linear(hidden_size, self.head_size * 2)
        self.linear2 = nn.Linear(self.head_size * 2, self.heads * 2)

        if self.use_rope:
            self.position_encoding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, hidden_state, mask=None):
        """ b: batch size
            c: ent_type_size
            l: sequence length
            h: hiddden size
            d: head_size
        """
        # hidden_states:  (b, l, h)
        sequence_output = self.linear1(hidden_state)  # (b, l, d * 2)

        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., heads, head_size]

        # RoPE编码 (乘性编码) 
        if self.use_rope:
            qw = self.position_encoding(qw)
            kw = self.position_encoding(kw)

        # 计算内积
        logits = torch.einsum('bmd, bnd->bmn', qw, kw)  # [btz, seq_len, seq_len]
        logits /= self.head_size ** 0.5

        bias_input = self.linear2(sequence_output)  # [..., heads*2]
        bias = torch.stack(
            torch.chunk(bias_input, self.heads, dim=-1), dim=-2
        ).transpose(1, 2)  # [btz, head_size, seq_len,2]
        # [btz, head_size, seq_len, seq_len]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)

        # 排除padding: method2
        if mask is not None:
            mask1 = 1 - mask[:, None, :, None]  # [btz, 1, seq_len, 1]
            mask2 = 1 - mask[:, None, None, :]  # [btz, 1, 1, seq_len]

            logits = logits.masked_fill(mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tri_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class Biaffine(nn.Module):
    """双仿射网络 Named Entity Recognition as Dependency Parsing
    """

    def __init__(self, hidden_size, head_size, num_labels, max_len=512,
                 add_position=True, bias=True, tri_mask=True):
        super(Biaffine, self).__init__()
        # use absolute position encoding
        self.add_position = add_position
        self.bias = bias
        self.tri_mask = tri_mask
        if add_position:
            self.position_encoding = SinusoidalPositionEncoding(max_len, hidden_size)

        # add lstm layers
        self.lstm = torch.nn.LSTM(
            hidden_size, hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )

        # 头预测层
        self.start_layer = nn.Sequential(nn.Linear(2 * hidden_size, head_size), nn.ReLU())
        # 尾预测层
        self.end_layer = nn.Sequential(nn.Linear(2 * hidden_size, head_size), nn.ReLU())

        self.u = nn.Parameter(torch.Tensor(head_size + self.bias, num_labels, head_size + self.bias))
        self.w = nn.Parameter(torch.Tensor(2 * (head_size + self.bias) + 1, num_labels))

        self.init_weights()

    def init_weights(self):
        """ 参数初始化 """
        torch.nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, hidden_state, mask=None):
        # hidden_state = [batch size, seq_len, hidden_size]
        seq_len = hidden_state.shape[1]

        if self.add_position:
            # 由于为加性拼接，我们无法使用RoPE,因此这里直接使用绝对位置编码
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_state.device)
            position_ids = position_ids.unsqueeze(0).expand_as(hidden_state)
            hidden_state += self.position_encoding(position_ids)

        hidden_state, _ = self.lstm(hidden_state)
        start_logits = self.start_layer(hidden_state)  # [b, l, d]
        end_logits = self.end_layer(hidden_state)

        if self.bias:
            # [b, l, d + 1]
            start_logits = torch.cat((start_logits, torch.ones_like(start_logits[..., :1])), dim=-1)
            end_logits = torch.cat((end_logits, torch.ones_like(end_logits[..., :1])), dim=-1)

        start_logits_con = torch.unsqueeze(start_logits, 1)  # [b, 1, l, d + 1]
        end_logits_con = torch.unsqueeze(end_logits, 2)  # [b, l, 1, d + 1]

        start_logits_con = start_logits_con.repeat(1, seq_len, 1, 1)  # [b, l, l, d + 1]
        end_logits_con = end_logits_con.repeat(1, 1, seq_len, 1)

        # [b, l, l, 2(d + 1)]
        concat_start_end = torch.cat([start_logits_con, end_logits_con], dim=-1)
        # [b, l, l, 2(d + 1) + 1]
        concat_start_end = torch.cat([concat_start_end, torch.ones_like(concat_start_end[..., :1])], dim=-1)

        # bix, xny, bjy -> bijn: [b, l, l, n]
        logits_1 = torch.einsum('bix, xny, bjy -> bijn', start_logits, self.u, end_logits)
        logits_2 = torch.einsum('bijy, yn -> bijn', concat_start_end, self.w)

        logits = logits_1 + logits_2
        logits = logits.permute(0, 3, 1, 2)  # [b, n, l, l]

        # 排除padding: method2
        if mask is not None:
            mask1 = 1 - mask[:, None, :, None]  # [btz, 1, seq_len, 1]
            mask2 = 1 - mask[:, None, None, :]  # [btz, 1, 1, seq_len]

            logits = logits.masked_fill(mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tri_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class UnlabeledEntity(nn.Module):
    """ https://arxiv.org/pdf/2012.05426.pdf
    """

    def __init__(self, hidden_size, num_labels, max_len=512, position_type='relative', tri_mask=True):
        super(UnlabeledEntity, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.position_type = position_type
        self.tri_mask = tri_mask

        self.Wh = nn.Linear(hidden_size * 4, self.hidden_size)
        self.Wo = nn.Linear(self.hidden_size, self.num_labels)

        # 位置编码
        if self.position_type == 'absolute':
            self.position_encoding = SinusoidalPositionEncoding(max_len, hidden_size)
        elif self.position_type == 'relative':
            self.relative_positions_encoding = RelativePositionsEncoding(
                qlen=max_len,
                klen=max_len,
                embedding_size=hidden_size * 4,
            )

    def forward(self, hidden_state, mask=None):
        seq_len = hidden_state.shape[1]

        # 绝对位置编码
        if self.position_type == 'absolute':
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_state.device)
            position_ids = position_ids.unsqueeze(0).expand_as(hidden_state)
            hidden_state += self.position_encoding(position_ids)

        start_logits = torch.unsqueeze(hidden_state, 1)  # [b, 1, l, h]
        end_logits = torch.unsqueeze(hidden_state, 2)  # [b, l, 1, h]

        start_logits = start_logits.repeat(1, seq_len, 1, 1)  # [b, l, l, h]
        end_logits = end_logits.repeat(1, 1, seq_len, 1)

        concat_inputs = torch.cat(
            [end_logits, start_logits, end_logits - start_logits, end_logits.mul(start_logits)], dim=-1
        )

        if self.position_type == 'relative':
            relations_keys = self.relative_positions_encoding(seq_len, seq_len)
            concat_inputs += relations_keys

        hij = torch.tanh(self.Wh(concat_inputs))
        logits = self.Wo(hij)  # [b, l, l, n]

        logits = logits.permute(0, 3, 1, 2)  # [b, n, l, l]

        # 排除padding: method2
        if mask is not None:
            mask1 = 1 - mask[:, None, :, None]  # [btz, 1, seq_len, 1]
            mask2 = 1 - mask[:, None, None, :]  # [btz, 1, 1, seq_len]

            logits = logits.masked_fill(mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tri_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class HandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type="cln"):
        super().__init__()
        self.shaking_type = shaking_type

        if shaking_type == "cat":
            self.cat_fc = nn.Linear(hidden_size * 2, hidden_size)
        if shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)

    def upper_reg2seq(self, tensor):
        """
        drop lower region and flat upper region to sequence
        :param tensor: (batch_size, matrix_size, matrix_size, hidden_size)
        :return: (batch_size, matrix_size + ... + 1, hidden_size)
        """
        bs, matrix_size, matrix_size, hidden_size = tensor.shape
        mask = torch.ones(matrix_size, matrix_size, device=tensor.device).triu().bool()[None, :, :, None]
        return tensor.masked_select(mask).reshape(bs, -1, hidden_size)

    def forward(self, seq_hiddens):
        """
        seq_hiddens: (batch_size, seq_len, hidden_size_x)
        return:
            if only look after:
                shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size); e.g. (32, 5+4+3+2+1, 5)
            else:
                shaking_hiddenss: (batch_size, seq_len * seq_len, hidden_size)
        """
        seq_len = seq_hiddens.size(1)

        guide = seq_hiddens[:, :, None, :].repeat(1, 1, seq_len, 1)
        visible = guide.permute(0, 2, 1, 3)

        # guide, visible: (batch_size, shaking_seq_len, hidden_size)
        guide = self.upper_reg2seq(guide)
        visible = self.upper_reg2seq(visible)

        if self.shaking_type == "cat":
            tp_cat_pre = torch.cat([guide, visible], dim=-1)
            shaking_pre = torch.relu(self.cat_fc(tp_cat_pre))

        if self.shaking_type == "cln":
            shaking_pre = self.tp_cln([visible, guide])

        return shaking_pre

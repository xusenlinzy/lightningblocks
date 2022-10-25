import torch
import torch.nn as nn


class MaxPool(nn.Module):
    def __init__(self, stride=None, padding=0, dilation=1, dimension=1, kernel_size=None, ceil_mode=False):
        """
        stride: 窗口移动大小，默认为kernel_size
        padding: padding的内容，默认为0
        dilation: 控制窗口内元素移动距离的大小
        dimension: MaxPool的维度，支持1，2，3维。
        kernel_size: max pooling的窗口大小，默认为tensor最后k维，其中k为dimension
        """
        super(MaxPool, self).__init__()
        assert dimension in [1, 2, 3], 'Now we only support 1d, 2d, or 3d Pooling'
        self.dimension = dimension
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.ceil_mode = ceil_mode

    def forward(self, x):
        if self.dimension == 1:
            x = torch.transpose(x, 1, 2)  # [N,L,C] -> [N,C,L]
            pooling = nn.MaxPool1d(
                stride=self.stride, padding=self.padding, dilation=self.dilation,
                kernel_size=self.kernel_size if self.kernel_size is not None else x.size(-1),
                return_indices=False, ceil_mode=self.ceil_mode
            )
        elif self.dimension == 2:
            pooling = nn.MaxPool2d(
                stride=self.stride, padding=self.padding, dilation=self.dilation,
                kernel_size=self.kernel_size if self.kernel_size is not None else (x.size(-2), x.size(-1)),
                return_indices=False, ceil_mode=self.ceil_mode
            )
        else:
            pooling = nn.MaxPool3d(
                stride=self.stride, padding=self.padding, dilation=self.dilation,
                kernel_size=self.kernel_size if self.kernel_size is not None else (x.size(-3), x.size(-2), x.size(-1)),
                return_indices=False, ceil_mode=self.ceil_mode
            )
        x = pooling(x)
        return x.squeeze(dim=-1)  # [N,C,1] -> [N,C]


class MaxPoolWithMask(nn.Module):
    """
    带mask矩阵的max pooling，在做max-pooling的时候不会考虑mask值为0的位置
    """

    def __init__(self):
        super(MaxPoolWithMask, self).__init__()
        self.inf = 10e12

    def forward(self, tensor, mask, dim=1):
        r"""
        :param torch.FloatTensor tensor: [batch_size, seq_len, channels] 初始tensor
        :param torch.LongTensor mask: [batch_size, seq_len] 0/1的mask矩阵
        :param int dim: 需要进行max pooling的维度
        :return:
        """
        masks = mask.view(mask.size(0), mask.size(1), -1)
        masks = masks.expand(-1, -1, tensor.size(2)).float()
        return torch.max(tensor + masks.le(0.5).float() * -self.inf, dim=dim)[0]


class KMaxPool(nn.Module):
    """K max-pooling module."""

    def __init__(self, k=1):
        super(KMaxPool, self).__init__()
        self.k = k

    def forward(self, x):
        r"""
        :param torch.Tensor x: [N, C, L] 初始tensor
        :return: torch.Tensor x: [N, C*k] k-max pool后的结果
        """
        x, index = torch.topk(x, self.k, dim=-1, sorted=False)
        x = torch.reshape(x, (x.size(0), -1))
        return x


class AvgPool(nn.Module):
    r"""
    给定形如[batch_size, max_len, hidden_size]的输入，在最后一维进行avg pooling. 输出为[batch_size, hidden_size]
    """

    def __init__(self, stride=None, padding=0):
        super(AvgPool, self).__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        r"""
        :param torch.Tensor x: [N, C, L] 初始tensor
        :return: torch.Tensor x: [N, C] avg pool后的结果
        """
        # [N,C,L] -> [N,C]
        kernel_size = x.size(2)
        pooling = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding)
        x = pooling(x)
        return x.squeeze(dim=-1)


class AvgPoolWithMask(nn.Module):
    r"""
    给定形如[batch_size, max_len, hidden_size]的输入，在最后一维进行avg pooling. 输出为[batch_size, hidden_size], pooling
    的时候只会考虑mask为1的位置
    """

    def __init__(self):
        super(AvgPoolWithMask, self).__init__()
        self.inf = 10e12

    def forward(self, tensor, mask, dim=1):
        r"""
        :param torch.FloatTensor tensor: [batch_size, seq_len, channels] 初始tensor
        :param torch.LongTensor mask: [batch_size, seq_len] 0/1的mask矩阵
        :param int dim: 需要进行max pooling的维度
        :return:
        """
        masks = mask.view(mask.size(0), mask.size(1), -1).float()
        return torch.sum(tensor * masks.float(), dim=dim) / torch.sum(masks.float(), dim=1)


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], f"unrecognized pooling type {self.pooler_type}"

    def forward(self, outputs, attention_mask):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

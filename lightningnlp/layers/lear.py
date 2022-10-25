import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelFusionForToken(nn.Module):
    def __init__(self, hidden_size):
        super(LabelFusionForToken, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, token_features, label_features, label_mask=None):
        bs, seqlen = token_features.shape[:2]
        token_features = self.linear1(token_features)  # [bs, seqlen, hidden_size]
        label_features = self.linear2(label_features)  # [c, label_len, hidden_size]

        # 计算注意力得分
        scores = torch.einsum('bmh, cnh->bmcn', token_features, label_features)
        scores += (1.0 - label_mask[None, None, ...]) * -10000.0
        scores = torch.softmax(scores, dim=-1)

        # 加权标签嵌入
        weighted_label_features = label_features[None, None, ...].repeat(bs, seqlen, 1, 1, 1) * scores[..., None]
        fused_features = token_features.unsqueeze(2) + weighted_label_features.sum(-2)
        return torch.tanh(self.output(fused_features))


class Classifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_labels, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(num_labels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        x = torch.mul(input, self.weight)
        x = torch.sum(x, -1)
        return x + self.bias


class MLPForMultiLabel(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_rate=0.2):
        super(MLPForMultiLabel, self).__init__()
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = Classifier(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, features):
        features = self.classifier1(features)
        features = self.dropout(F.gelu(features))
        return self.classifier2(features)

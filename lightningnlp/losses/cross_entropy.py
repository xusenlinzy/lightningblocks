import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, preds, target):
        loss = torch.sum(-target * F.log_softmax(preds, dim=-1), dim=-1)
        return loss.mean()


class MultiLabelCategoricalCrossEntropy(nn.Module):
    """
    https://kexue.fm/archives/7359
    """

    def __init__(self):
        super(MultiLabelCategoricalCrossEntropy, self).__init__()

    def forward(self, preds, target):
        preds = (1 - 2 * target) * preds  # -1 -> pos classes, 1 -> neg classes
        preds_neg = preds - target * 1e12  # mask the pred outputs of pos classes
        preds_pos = (preds - (1 - target) * 1e12)  # mask the pred outputs of neg classes

        zeros = torch.zeros_like(preds[..., :1])
        preds_neg = torch.cat([preds_neg, zeros], dim=-1)
        preds_pos = torch.cat([preds_pos, zeros], dim=-1)

        neg_loss = torch.logsumexp(preds_neg, dim=-1)
        pos_loss = torch.logsumexp(preds_pos, dim=-1)
        return (neg_loss + pos_loss).mean()


class SparseMultilabelCategoricalCrossentropy(nn.Module):
    """稀疏版多标签分类的交叉熵
    https://kexue.fm/archives/8888
    """

    def __init__(self, mask_zero=False):
        super(SparseMultilabelCategoricalCrossentropy, self).__init__()
        self.mask_zero = mask_zero

    def forward(self, preds, target):
        """
        preds.shape: [bsz, heads, seq_len * seq_len]
        target.shape: # [bsz, heads, num_spoes]
        """

        zeros = torch.zeros_like(preds[..., :1])
        preds = torch.cat([preds, zeros], dim=-1)  # [0,1,2,...,l*l-1,0]

        if self.mask_zero:
            infs = zeros + 1e12
            preds = torch.cat([infs, preds[..., 1:]], dim=-1)

        pos_2 = torch.gather(preds, dim=-1, index=target)  # [bsz, heads, num_spoes]
        pos_1 = torch.cat([pos_2, zeros], dim=-1)

        if self.mask_zero:
            preds = torch.cat([-infs, preds[..., 1:]], dim=-1)
            pos_2 = torch.gather(preds, dim=-1, index=target)

        pos_loss = torch.logsumexp(-pos_1, axis=-1)
        all_loss = torch.logsumexp(preds, axis=-1)
        aux_loss = torch.logsumexp(pos_2, axis=-1) - all_loss
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=1e-7, max=1)
        neg_loss = all_loss + torch.log(aux_loss)
        return pos_loss + neg_loss

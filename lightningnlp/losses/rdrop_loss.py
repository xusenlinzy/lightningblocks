import torch
import torch.nn as nn
import torch.nn.functional as F


class RDropLoss(nn.Module):
    """
    R-Drop Loss implementation
    For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
    Original implementation please refer to this code: https://github.com/dropreg/R-Drop

    Args:
        reduction(str, optional):
            Indicate how to average the loss, the candicates are ``'none'``,``'batchmean'``,``'mean'``,``'sum'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
            If `reduction` is ``'sum'``, the reduced sum loss is returned;
            If `reduction` is ``'none'``, no reduction will be applied.
            Defaults to ``'none'``.
    """

    def __init__(self, reduction='none'):
        super(RDropLoss, self).__init__()
        if reduction not in ['sum', 'mean', 'none', 'batchmean']:
            raise ValueError(f"'reduction' in 'RDropLoss' should be 'sum', 'mean' 'batchmean', or 'none', but received {reduction}.")

        self.reduction = reduction

    def forward(self, p, q, pad_mask=None):
        """
        Args:
            p(Tensor): the first forward logits of training examples.
            q(Tensor): the second forward logits of training examples.
            pad_mask(Tensor, optional): The Tensor containing the binary mask to index with, it's data type is bool.

        Returns:
            Tensor: Returns tensor `loss`, the rdrop loss of p and q.
        """
        p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                          F.softmax(q, dim=-1),
                          reduction=self.reduction)
        q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                          F.softmax(p, dim=-1),
                          reduction=self.reduction)

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss = torch.masked_select(p_loss, pad_mask)
            q_loss = torch.masked_select(q_loss, pad_mask)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        return (p_loss + q_loss) / 2
        
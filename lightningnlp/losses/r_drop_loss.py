import torch
import torch.nn as nn
import torch.nn.functional as F


class RDropLoss(nn.Module):
    """ R-Drop的Loss实现，官方项目：https://github.com/dropreg/R-Drop
    """
    def __init__(self, alpha=4, rank='adjacent'):
        super().__init__()
        self.alpha = alpha
        assert rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rank = rank
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, y_pred, y_true):
        """支持两种方式，一种是奇偶相邻排列，一种是上下排列
        """
        loss_sup = self.loss_sup(y_pred, y_true)

        if self.rank == 'adjacent':
            y_pred1 = y_pred[1::2]
            y_pred2 = y_pred[::2]
        elif self.rank == 'updown':
            half_btz = y_true.shape[0] // 2
            y_pred1 = y_pred[:half_btz]
            y_pred2 = y_pred[half_btz:]
        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))
        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha

        
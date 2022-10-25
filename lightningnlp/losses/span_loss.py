import torch
import torch.nn as nn
from .focal_loss import FocalLoss
from .label_smoothing import LabelSmoothingCE


class SpanLoss(nn.Module):
    def __init__(self, num_labels, loss_type='cross_entropy', reduction='mean'):
        super().__init__()
        assert loss_type in ['cross_entropy', 'focal_loss', 'label_smoothing_ce']
        loss_fcts = {
            'cross_entropy': nn.CrossEntropyLoss(reduction=reduction),
            'focal_loss': FocalLoss(num_labels, reduction=reduction),
            'label_smoothing_ce': LabelSmoothingCE(reduction=reduction)
        }
        self.loss_fct = loss_fcts[loss_type]

    def forward(self, preds, target, masks):
        # assert if inp and target has both start and end values
        assert len(preds) == 2, "start and end logits should be present for spn losses calc"
        assert len(target) == 2, "start and end logits should be present for spn losses calc"
        assert masks is not None, "masks should be provided."

        active_loss = masks.view(-1) == 1
        start_logits, end_logits = preds
        start_positions, end_positions = target

        start_logits = start_logits.view(-1, start_logits.size(-1))
        end_logits = end_logits.view(-1, start_logits.size(-1))

        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]
        active_start_labels = start_positions.view(-1)[active_loss]
        active_end_labels = end_positions.view(-1)[active_loss]

        start_loss = self.loss_fct(active_start_logits, active_start_labels)
        end_loss = self.loss_fct(active_end_logits, active_end_labels)
        return (start_loss + end_loss) / 2


class SpanLossForMultiLabel(nn.Module):
    def __init__(self, name='Span Binary Cross Entropy Loss'):
        super().__init__()
        self.name = name
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, target, masks, nested=False):
        assert masks is not None, "masks should be provided."
        if not nested:
            return self.flated_forward(preds, target, masks)
        start_logits, end_logits, span_logits = preds
        start_labels, end_labels, span_labels = target
        start_, end_ = start_logits > 0, end_logits > 0

        bs, seqlen, num_labels = start_logits.shape
        span_candidate = torch.logical_or(
            (start_.unsqueeze(-2).expand(-1, -1, seqlen, -1) & end_.unsqueeze(-3).expand(-1, seqlen, -1, -1)),
            (start_labels.unsqueeze(-2).expand(-1, -1, seqlen, -1).bool() & end_labels.unsqueeze(-3).expand(-1, seqlen,
                                                                                                            -1,
                                                                                                            -1).bool())
        )

        masks = masks[:, :, None].expand(-1, -1, num_labels)
        start_loss = self.loss_fct(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * masks.reshape(-1)).view(-1, num_labels).sum(-1).sum() / (masks.sum() / num_labels)

        end_loss = self.loss_fct(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * masks.reshape(-1)).view(-1, num_labels).sum(-1).sum() / (masks.sum() / num_labels)

        span_masks = masks.bool().unsqueeze(2).expand(-1, -1, seqlen, -1) & masks.bool().unsqueeze(1).expand(-1, seqlen,
                                                                                                             -1, -1)
        span_masks = torch.triu(span_masks.permute(0, 3, 1, 2), 0).permute(0, 2, 3,
                                                                           1) * span_candidate  # start should be less equal to end
        span_loss = self.loss_fct(span_logits.view(bs, -1), span_labels.view(bs, -1).float())
        span_loss = span_loss.reshape(-1, num_labels).sum(-1).sum() / (
                    span_masks.view(-1, num_labels).sum() / num_labels)

        return start_loss + end_loss + span_loss

    def flated_forward(self, preds, target, masks):
        active_loss = masks.view(-1) == 1
        start_logits, end_logits = preds
        start_labels, end_labels = target

        active_start_logits = start_logits.view(-1, start_logits.size(-1))[active_loss]
        active_end_logits = end_logits.view(-1, start_logits.size(-1))[active_loss]

        active_start_labels = start_labels.view(-1, start_labels.size(-1))[active_loss].float()
        active_end_labels = end_labels.view(-1, end_labels.size(-1))[active_loss].float()

        start_loss = self.loss_fct(active_start_logits, active_start_labels).sum(1).mean()
        end_loss = self.loss_fct(active_end_logits, active_end_labels).sum(1).mean()
        return start_loss + end_loss

import torch
import torch.nn as nn
import torch.nn.functional as F


def cumsoftmax(x):
    return torch.cumsum(F.softmax(x,-1),dim=-1)


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.bool
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if not self.training:
            return F.linear(input, self.weight * (1 - self.dropout), self.bias)
        if sample_mask:
            self.sample_mask()
        return F.linear(input, self._weight, self.bias)


class pfn_unit(nn.Module):
    def __init__(self, hidden_size, input_size, dropconnect=0.1):
        super(pfn_unit, self).__init__()
        self.hidden_transform = LinearDropConnect(hidden_size, 5 * hidden_size, bias=True, dropout=dropconnect)
        self.input_transform = nn.Linear(input_size, 5 * hidden_size, bias=True)

        self.transform = nn.Linear(hidden_size * 3, hidden_size)
        self.drop_weight_modules = [self.hidden_transform]

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


    def forward(self, x, hidden):
        h_in, c_in = hidden

        gates = self.input_transform(x) + self.hidden_transform(h_in)
        c, eg_cin, rg_cin, eg_c, rg_c = gates[:, :].chunk(5, 1)

        eg_cin = 1 - cumsoftmax(eg_cin)
        rg_cin = cumsoftmax(rg_cin)

        eg_c = 1 - cumsoftmax(eg_c)
        rg_c = cumsoftmax(rg_c)

        c = torch.tanh(c)

        overlap_c = rg_c * eg_c
        upper_c = rg_c - overlap_c
        downer_c = eg_c - overlap_c

        overlap_cin =rg_cin * eg_cin
        upper_cin = rg_cin - overlap_cin
        downer_cin = eg_cin - overlap_cin

        share = overlap_cin * c_in + overlap_c * c

        c_re = upper_cin * c_in + upper_c * c + share
        c_ner = downer_cin * c_in + downer_c * c + share
        c_share = share

        h_re = torch.tanh(c_re)
        h_ner = torch.tanh(c_ner)
        h_share = torch.tanh(c_share)
        
        c_out = torch.cat((c_re, c_ner, c_share), dim=-1)
        c_out = self.transform(c_out)
        h_out = torch.tanh(c_out)

        return (h_out, c_out), (h_ner, h_re, h_share)


class encoder(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size
        self.unit = pfn_unit(hidden_size, input_size)

    def hidden_init(self, features):
        batch_size = features.size(1)
        h0 = torch.zeros(batch_size, self.hidden_size).requires_grad_(False).to(features.device)
        c0 = torch.zeros(batch_size, self.hidden_size).requires_grad_(False).to(features.device)
        return (h0, c0)

    def forward(self, features):
        seq_len = features.size(0)
        h_ner, h_re, h_share = [], [], []
        hidden = self.hidden_init(features)

        if self.training:
            self.unit.sample_masks()

        for t in range(seq_len):
            hidden, h_task = self.unit(features[t, :, :], hidden)
            h_ner.append(h_task[0])
            h_re.append(h_task[1])
            h_share.append(h_task[2])

        h_ner = torch.stack(h_ner, dim=0)
        h_re = torch.stack(h_re, dim=0)
        h_share = torch.stack(h_share, dim=0)

        return h_ner, h_re, h_share


class ner_unit(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super(ner_unit, self).__init__()
        self.num_labels = num_labels
        self.hid2hid = nn.Linear(hidden_size * 3, hidden_size)
        self.hid2tag = nn.Linear(hidden_size, num_labels)

        self.elu = nn.ELU()
        self.n = nn.Linear(hidden_size * 2, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_ner, h_share, mask):
        length, batch_size, _ = h_ner.size()
        h_global = torch.cat((h_share, h_ner), dim=-1)
        h_global = torch.tanh(self.n(h_global))

        h_global = torch.max(h_global, dim=0)[0]
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1, 1)

        st = h_ner.unsqueeze(1).repeat(1, length, 1, 1)
        en = h_ner.unsqueeze(0).repeat(length, 1, 1, 1)

        ner = torch.cat((st, en, h_global), dim=-1)
        ner = self.ln(self.hid2hid(ner))
        ner = self.elu(self.dropout(ner))
        ner = torch.sigmoid(self.hid2tag(ner))

        diagonal_mask = torch.triu(torch.ones(batch_size, length, length)).to(h_ner.device)
        diagonal_mask = diagonal_mask.permute(1, 2, 0)

        mask_s = mask.unsqueeze(1).repeat(1, length, 1)
        mask_e = mask.unsqueeze(0).repeat(length, 1, 1)

        mask_ner = mask_s * mask_e * diagonal_mask
        mask_ner = mask_ner.unsqueeze(-1).repeat(1, 1, 1, self.num_labels)

        ner = ner * mask_ner

        return ner


class re_unit(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super(re_unit, self).__init__()
        self.num_labels = num_labels
        self.hid2hid = nn.Linear(hidden_size * 3, hidden_size)
        self.hid2rel = nn.Linear(hidden_size, num_labels)

        self.elu = nn.ELU()
        self.r = nn.Linear(hidden_size * 2, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_re, h_share, mask):
        length = h_re.size(0)

        h_global = torch.cat((h_share, h_re), dim=-1)
        h_global = torch.tanh(self.r(h_global))

        h_global = torch.max(h_global, dim=0)[0]
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1, 1)

        r1 = h_re.unsqueeze(1).repeat(1, length, 1, 1)
        r2 = h_re.unsqueeze(0).repeat(length, 1, 1, 1)

        re = torch.cat((r1, r2, h_global), dim=-1)
        re = self.ln(self.hid2hid(re))
        re = self.elu(self.dropout(re))
        re = torch.sigmoid(self.hid2rel(re))

        mask = mask.unsqueeze(-1).repeat(1, 1, self.num_labels)
        mask_e1 = mask.unsqueeze(1).repeat(1, length, 1, 1)
        mask_e2 = mask.unsqueeze(0).repeat(length, 1, 1, 1)
        mask_re = mask_e1 * mask_e2

        re = re * mask_re

        return re
    
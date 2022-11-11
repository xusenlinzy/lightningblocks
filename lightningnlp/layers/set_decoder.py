import torch
import torch.nn as nn

from .transformer import TransformerDecoderLayer


class SetDecoder(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers,
                 num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate

        self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(num_layers)])

        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.decoder2class = nn.Linear(config.hidden_size, num_classes + 1)

        self.head_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)

        self.tail_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)

        self.head_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.tail_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.head_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)

        self.tail_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)

        torch.nn.init.orthogonal_(self.head_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None):
        bsz = encoder_hidden_states.size()[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        if hidden_states is not None:
            hidden_states = hidden_states + query_embed
            hidden_states = self.dropout(self.layernorm(hidden_states))
        else:
            hidden_states = self.dropout(self.layernorm(query_embed))

        all_hidden_states = ()
        for layer_module in self.layers:
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]  # [B, N, H]

        class_logits = self.decoder2class(hidden_states)  # [B, N, R + 1]

        head_start_logits = self.head_start_metric_3(torch.tanh(
            self.head_start_metric_1(hidden_states).unsqueeze(2) + self.head_start_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        head_end_logits = self.head_end_metric_3(torch.tanh(
            self.head_end_metric_1(hidden_states).unsqueeze(2) + self.head_end_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        tail_start_logits = self.tail_start_metric_3(torch.tanh(
            self.tail_start_metric_1(hidden_states).unsqueeze(2) + self.tail_start_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        tail_end_logits = self.tail_end_metric_3(torch.tanh(
            self.tail_end_metric_1(hidden_states).unsqueeze(2) + self.tail_end_metric_2(
                encoder_hidden_states).unsqueeze(1))).squeeze()

        return class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits

from typing import Optional

import torch
import torch.nn as nn
from transformers import ErnieModel, ErniePreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import MultipleChoiceModelOutput


class UTC(ErniePreTrainedModel):
    """
    Ernie Model with two linear layer on the top of the hidden-states output to compute
    probability of candidate labels, designed for Unified Tag Classification.
    """

    def __init__(self, config: PretrainedConfig):
        super(UTC, self).__init__(config)
        self.encoder = ErnieModel(config)
        self.predict_size = 64
        self.linear_q = nn.Linear(config.hidden_size, self.predict_size)
        self.linear_k = nn.Linear(config.hidden_size, self.predict_size)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        omask_positions: Optional[torch.Tensor] = None,
        cls_positions: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor):
                See :class:`ErnieModel`.
            position_ids (Tensor):
                See :class:`ErnieModel`.
            attention_mask (Tensor):
                See :class:`ErnieModel`.
            omask_positions (Tensor of shape `(batch_size, max_option)`):
                Masked positions of [O-MASK] tokens padded with 0.
            cls_positions (Tensor of shape `(batch_size)`):
                Masked positions of the second [CLS] token.
        """
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        batch_size, seq_len, hidden_size = sequence_output.shape
        flat_sequence_output = sequence_output.reshape(-1, hidden_size)

        cls_output = torch.gather(flat_sequence_output, 0, cls_positions.unsqueeze(-1).expand(-1, hidden_size))
        q = self.linear_q(cls_output)

        flat_length = torch.arange(batch_size) * seq_len
        flat_length = torch.tensor(flat_length.unsqueeze(dim=1), dtype=torch.int64)
        option_output = torch.gather(
            flat_sequence_output, 0, (omask_positions + flat_length).reshape(-1, 1).expand(-1, hidden_size)
        )
        option_output = option_output.reshape(batch_size, -1, hidden_size)
        k = self.linear_k(option_output)

        option_logits = torch.matmul(q.unsqueeze(1), k.T).squeeze(1)
        option_logits = option_logits / self.predict_size**0.5
        for index, logit in enumerate(option_logits):
            option_logits[index] -= (1 - (omask_positions[index] > 0).float()) * 1e12

        loss = None
        if not return_dict:
            output = (option_logits,)
            if output_hidden_states:
                output = output + (outputs.hidden_states,)
            if output_attentions:
                output = output + (outputs.attentions,)
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=option_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

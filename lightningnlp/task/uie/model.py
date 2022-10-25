import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import BertModel, BertPreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput
from typing import Optional, Tuple


@dataclass
class UIEModelOutput(ModelOutput):
    # noinspection PyUnresolvedReferences
    """
        Output class for outputs of UIE.
        Args:
            losses (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
                Total spn extraction losses is the sum of a Cross-Entropy for the start and end positions.
            start_prob (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Span-start scores (after Sigmoid).
            end_prob (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Span-end scores (after Sigmoid).
            hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layers, +
                one for the output of each layers) of shape `(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layers plus the optional initial embedding outputs.
            attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                Tuple of `torch.FloatTensor` (one for each layers) of shape `(batch_size, num_heads, sequence_length,
                sequence_length)`.
                Attention weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """
    loss: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    start_positions: torch.FloatTensor = None
    end_positions: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class UIE(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super(UIE, self).__init__(config)
        self.bert = BertModel(config)
        self.config = config

        self.linear_start = nn.Linear(config.hidden_size, 1)
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        if hasattr(config, 'use_task_id') and config.use_task_id:
            # Add task type embedding to BERT
            task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)
            self.bert.embeddings.task_type_embeddings = task_type_embeddings

            # noinspection PyUnusedLocal
            def hook(module, input, output):
                return output + task_type_embeddings(
                    torch.zeros(input[0].size(), dtype=torch.int64, device=input[0].device))

            self.bert.embeddings.word_embeddings.register_forward_hook(hook)

        self.post_init()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                ) -> UIEModelOutput:
        """
        Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled spn for computing the token classification losses.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the losses.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled spn for computing the token classification losses.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the losses.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]

        start_logits = self.linear_start(sequence_output).squeeze(-1)
        start_prob = self.sigmoid(start_logits)

        end_logits = self.linear_end(sequence_output).squeeze(-1)
        end_prob = self.sigmoid(end_logits)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.BCELoss()
            start_loss = loss_fct(start_prob, start_positions.float())
            end_loss = loss_fct(end_prob, end_positions.float())
            loss = (start_loss + end_loss) / 2.0

        return UIEModelOutput(
            loss=loss,
            start_prob=start_prob,
            end_prob=end_prob,
            start_positions=start_positions,
            end_positions=end_positions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class UIEM(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super(UIEM, self).__init__(config)
        self.bert = BertModel(config)
        self.config = config

        self.linear_start = nn.Linear(config.hidden_size, 1)
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        if hasattr(config, 'use_task_id') and config.use_task_id:
            # Add task type embedding to BERT
            task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)
            self.bert.embeddings.task_type_embeddings = task_type_embeddings

            # noinspection PyUnusedLocal
            def hook(module, input, output):
                return output + task_type_embeddings(
                    torch.zeros(input[0].size(), dtype=torch.int64, device=input[0].device))

            self.bert.embeddings.word_embeddings.register_forward_hook(hook)

        self.post_init()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                ) -> UIEModelOutput:
        """
        Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled spn for computing the token classification losses.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the losses.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled spn for computing the token classification losses.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the losses.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        outputs = self.bert(
            input_ids=input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]

        start_logits = self.linear_start(sequence_output).squeeze(-1)
        start_prob = self.sigmoid(start_logits)

        end_logits = self.linear_end(sequence_output).squeeze(-1)
        end_prob = self.sigmoid(end_logits)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.BCELoss()
            start_loss = loss_fct(start_prob, start_positions.float())
            end_loss = loss_fct(end_prob, end_positions.float())
            loss = (start_loss + end_loss) / 2.0

        return UIEModelOutput(
            loss=loss,
            start_prob=start_prob,
            end_prob=end_prob,
            start_positions=start_positions,
            end_positions=end_positions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

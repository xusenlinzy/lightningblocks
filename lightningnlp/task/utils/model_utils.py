from dataclasses import dataclass
from typing import *

import torch
from transformers import BertModel, BertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.models.albert.modeling_albert import AlbertModel, AlbertPreTrainedModel
from transformers.models.ernie.modeling_ernie import ErnieModel, ErniePreTrainedModel
from transformers.models.nezha.modeling_nezha import NezhaModel, NezhaPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import XLNetModel, XLNetPreTrainedModel

from lightningnlp.models import ChineseBertModel
from lightningnlp.models import RoFormerModel, RoFormerPreTrainedModel

MODEL_MAP = {
    "bert": (BertModel, BertPreTrainedModel),
    "ernie": (ErnieModel, ErniePreTrainedModel),
    "roformer": (RoFormerModel, RoFormerPreTrainedModel),
    "nezha": (NezhaModel, NezhaPreTrainedModel),
    "albert": (AlbertModel, AlbertPreTrainedModel),
    "xlnet": (XLNetModel, XLNetPreTrainedModel),
    "chinese-bert": (ChineseBertModel, BertPreTrainedModel),
}


@dataclass
class SequenceLabelingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: List[Any] = None
    groundtruths: List[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RelationExtractionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    predictions: List[Any] = None
    groundtruths: List[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SpanOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    span_logits: Optional[torch.FloatTensor] = None
    predictions: List[Any] = None
    groundtruths: List[Any] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SiameseClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


@dataclass
class SentenceEmbeddingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[torch.FloatTensor] = None


@dataclass
class UIEModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

import torch
import torch.nn as nn
from .layer_norm import LayerNorm


class WordEmbedding(nn.Module):
    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.layer_norm = nn.LayerNorm(args.emb_size)

    def forward(self, src, _):
        emb = self.word_embedding(src)
        emb = self.dropout(self.layer_norm(emb))
        return emb


class BertEmbeddings(nn.Module):
    """
        embeddings层
        构造word, position and token_type embeddings.
    """

    def __init__(
            self,
            vocab_size,
            embedding_size,
            hidden_size,
            max_position,
            segment_vocab_size,
            shared_segment_embeddings,
            drop_rate,
            conditional_size=False,
            **kwargs
    ):
        super(BertEmbeddings, self).__init__()
        self.shared_segment_embeddings = shared_segment_embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        if (kwargs.get('p_bias') not in {'rotary', 'typical_relative',
                                         't5_relative'}) and max_position > 0:  # Embeddings时候包含位置编码
            self.position_embeddings = nn.Embedding(max_position, embedding_size)
        if (segment_vocab_size > 0) and (not shared_segment_embeddings):
            self.segment_embeddings = nn.Embedding(segment_vocab_size, embedding_size)

        self.layerNorm = LayerNorm(embedding_size, eps=1e-12, conditional_size=conditional_size)
        self.dropout = nn.Dropout(drop_rate)
        # 如果embedding_size != hidden_size，则再有一个linear(适用于albert矩阵分解)
        if embedding_size != hidden_size:
            self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)

    def forward(self,
                token_ids,
                segment_ids=None,
                conditional_emb=None):
        words_embeddings = self.word_embeddings(token_ids)

        if hasattr(self, 'segment_embeddings'):
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.segment_embeddings(segment_ids)
            embeddings = words_embeddings + segment_embeddings
        elif self.shared_segment_embeddings:  # segment和word_embedding共享权重
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.word_embeddings(segment_ids)
            embeddings = words_embeddings + segment_embeddings
        else:
            embeddings = words_embeddings

        if hasattr(self, 'position_embeddings'):
            seq_length = token_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if hasattr(self, 'layerNorm'):
            embeddings = self.layerNorm([embeddings, conditional_emb])
        embeddings = self.dropout(embeddings)

        if hasattr(self, 'embedding_hidden_mapping_in'):
            embeddings = self.embedding_hidden_mapping_in(embeddings)
        return embeddings

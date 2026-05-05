# Copyright (c) 2022, Tri Dao.

""""
The implementation was adopted from
https://github.com/Dao-AILab/flash-attention/blob/43950dda456e095969d842fca7a73c5bfe3cecd0/flash_attn/models/bert.py
"""

import torch
import torch.nn as nn
from torch import Tensor


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        type_vocab_size,
        padding_idx=None,
        device=None,
        dtype=None,
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If type_vocab_size <= 0, there's no token type embeddings
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs
        )
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim, **factory_kwargs
            )
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(type_vocab_size, embed_dim, **factory_kwargs)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        token_type_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(seqlen, dtype=torch.long, device=input_ids.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
        return embeddings

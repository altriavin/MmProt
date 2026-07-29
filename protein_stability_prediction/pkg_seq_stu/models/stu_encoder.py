from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
# from .TransformerBlock import ResidualAttentionBlock


import torch
import torch.nn as nn
import math

# from torchdrug import models
# from .TransformerBlock import MLP, LayerNorm
from transformers import EsmTokenizer, EsmForMaskedLM
from pkg_seq_stu.constants import esm_root_path, saport_root_path


class SaPort_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = EsmForMaskedLM.from_pretrained(
            saport_root_path, weights_only=False
        )

        for param in self.model.parameters():
            param.requires_grad = False 

    def forward(self, stu_seqs):
        stu_seqs = {k: v.cuda() for k, v in stu_seqs.items()}
        stu_seqs["output_hidden_states"] = True
        with torch.no_grad():
            saport_emb = self.model.esm(**stu_seqs)["hidden_states"][-1]
        return saport_emb


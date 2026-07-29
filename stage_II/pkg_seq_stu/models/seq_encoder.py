import torch
from torch import nn

from collections import OrderedDict

from transformers import EsmTokenizer, EsmForMaskedLM
from pkg_seq_stu.constants import esm_root_path, saport_root_path


class esm_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = EsmForMaskedLM.from_pretrained(
            esm_root_path, weights_only=False
        )

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, seq_seqs):
        seq_seqs = {k: v.cuda() for k, v in seq_seqs.items()}
        seq_seqs["output_hidden_states"] = True
        with torch.no_grad():
            seq_emb = self.model(**seq_seqs)["hidden_states"][-1]
        return seq_emb


class export_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = EsmForMaskedLM.from_pretrained(
            esm_root_path, weights_only=False
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, seq_seqs):


        
        seq_seqs = {k: v.cuda() for k, v in seq_seqs.items()}
        with torch.no_grad():
            seq_logits = self.model(**seq_seqs).logits
        return seq_logits


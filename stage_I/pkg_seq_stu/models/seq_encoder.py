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


class carp_encoder(nn.Module):
    def __init__(self, seq_model_dim=128):
        super().__init__()
        self.model, self.collater = load_model_and_alphabet('/kaiwu_vepfs_volc/kaiwu_vepfs_volc/huxiaowen/pretrain/carp_640M.pt')
        self.norm_in = nn.LayerNorm(1280, eps=1e-5)
        self.norm_out = nn.LayerNorm(seq_model_dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, seq_model_dim),
            self.norm_out
        )

    def forward(self, seqs):
        with torch.no_grad():
            x = self.collater(seqs)[0]
            mask = x.eq(27)
            rep = self.model(x)["representations"][56]
        rep = self.norm_in(rep)
        rep = self.mlp(rep)
        return rep, mask
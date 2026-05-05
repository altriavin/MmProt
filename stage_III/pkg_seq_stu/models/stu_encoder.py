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


class GearNet_encoder(nn.Module):
    def __init__(self, stu_model_dim, dropout):
        super().__init__()

        gearnet_pretrain_root_path = "/kaiwu_vepfs_volc/kaiwu_vepfs_volc/huxiaowen/pretrain/mc_gearnet_edge.pth"
        self.model = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512],
                                num_relation=7, edge_input_dim=59, num_angle_bin=8,
                                batch_norm=True, concat_hidden=True, short_cut=True, readout="mean")
        net = torch.load(gearnet_pretrain_root_path)
        self.model.load_state_dict(net)

        for param in self.model.parameters():
            param.requires_grad = False

        self.stu_ln_in = LayerNorm(3072)
        self.stu_align_head = MLP(indim=3072, hiddim=stu_model_dim * 4, outdim=stu_model_dim, dropout=True, dropoutp=dropout)
        self.stu_ln_out = LayerNorm(stu_model_dim)

    def forward(self, stu_grapps):
        output = self.model(stu_grapps, stu_grapps.node_feature.float(), all_loss=None, metric=None)
        print(f'output: {output}')
        gearnet_emb = output['node_feature']

        print(f'grarnet_emb.shape: {gearnet_emb.shape}')

        rep = self.stu_ln_in(gearnet_emb)
        rep = self.stu_align_head(rep)
        rep = self.stu_ln_out(rep)
        return rep


        # self.pos_embed = LearnedPositionalEmbedding(stu_model_dim)
        # self.stu_embed = nn.Embedding(2048 + 4 + 1, stu_model_dim)

        # self.blocks = nn.ModuleList(
        #     [ResidualAttentionBlock(d_model=stu_model_dim, n_head=stu_heads, add_bias_kv=False,
        #                                dropout=dropout, batch_first=True) for _ in range(stu_layers)]
        # )
        # self.ln_f = nn.LayerNorm(stu_model_dim)

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     key_padding_mask: Optional[torch.Tensor],
    # ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    #     B, T = x.shape
    #     x = self.stu_embed(x) + self.pos_embed(key_padding_mask)

    #     for blk in self.blocks:
    #         x = blk(x, key_padding_mask=key_padding_mask)
    #     x = self.ln_f(x)
    #     return x
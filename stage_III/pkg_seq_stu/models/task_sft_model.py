import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoderDecoderModel,
    FairseqEncoder,
    FairseqEncoderModel,
    FairseqDecoder,
    register_model,
    register_model_architecture,
)

from .fla_attn.modeling_bert import BertModel
from .seq_stu_model import seq_stu_align_encoder
from .seq_encoder import export_encoder
from pkg_seq_stu.constants import pretrain_root_path, AA_TO_IDX
import os

from einops import rearrange

class viral_model(FairseqEncoder):
    def __init__(
            self, args, dictionary
    ):
        super().__init__(dictionary)
        self.model = seq_stu_align_encoder(args, dictionary)
        self.pred_head = nn.Linear(args.hidden_size, 33)

    def forward(self, wt_seqs, wt_stus, **kwargs):
        # src_inputs, stu_inputs = inputs["src_inputs"], inputs["stu_inputs"]

        mlm_fusion = self.model.forward_inference(wt_seqs, wt_stus)
        mlm_z = mlm_fusion["encoder_out"]["z"]
        mlm_probs = self.pred_head(mlm_z)

        return {
            "encoder_out": {
                "mlm_probs": mlm_probs
            },
            "encoder_states": []
        }
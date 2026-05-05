from .stu_encoder import SaPort_encoder
from .seq_encoder import esm_encoder
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


class seq_stu_align_encoder(FairseqEncoder):
    def __init__(
            self, args, dictionary
    ):
        super().__init__(dictionary)
        self.stu_encoder = SaPort_encoder(args)
        self.stu_proj = nn.Linear(args.stu_model_dim, args.hidden_size)

        self.seq_encoder = esm_encoder(args)
        self.seq_proj = nn.Linear(args.seq_model_dim, args.hidden_size)

        self.fusion_encoder = BertModel(args)

        self.eps_seq = nn.Parameter(torch.ones(1, 1, args.hidden_size) * 0.2)
        self.eps_stu = nn.Parameter(torch.ones(1, 1, args.hidden_size) * 0.2)

    def data_argument_noisy(
            self,
            embeddings: torch.Tensor,
            eps: float = 0.15,
            masks: Optional[torch.Tensor] = None,
            rand_angle: bool = True,
            generator: Optional[torch.Generator] = None
    ):
        B, L, D = embeddings.shape
        device, dtype = embeddings.device, embeddings.dtype

        if masks is None:
            valid = torch.ones(B, L, dtype=torch.bool, device=device)
        else:
            valid = (~masks) if masks.dtype == torch.bool else masks.eq(0)

        e32 = embeddings.to(torch.float32)

        r = e32.norm(dim=-1, keepdim=True)
        ehat = e32 / r.clamp_min(1e-12)

        n = torch.randn_like(e32, dtype=torch.float32)
        t = n - (n * ehat).sum(dim=-1, keepdim=True) * ehat
        u = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        if rand_angle:
            theta = torch.rand(B, L, 1, device=device) * eps
        else:
            theta = torch.full((B, L, 1), fill_value=eps, device=device)

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        e_rot = (cos_t * ehat + sin_t * u) * r

        nonzero = (r > 1e-12)
        out32 = torch.where(nonzero, e_rot, e32)
        out32 = torch.where(valid.unsqueeze(-1), out32, e32)

        return out32.to(dtype)


    def forward(self, seq_seqs, aug_seq_seqs, stu_seqs, aug_stu_seqs, seq_masks, stu_masks, **kwargs):

        seq_x = self.seq_encoder(seq_seqs)
        seq_x_aug_mut = self.seq_encoder(aug_seq_seqs)
        seq_x_aug_noisy = self.data_argument_noisy(seq_x, masks=seq_masks)

        seq_x = self.seq_proj(seq_x)
        seq_x_aug_mut = self.seq_proj(seq_x_aug_mut)
        seq_x_aug_noisy = self.seq_proj(seq_x_aug_noisy)

        stu_x = self.stu_encoder(stu_seqs)
        stu_x_aug_mut = self.stu_encoder(aug_stu_seqs)
        stu_x_aug_noisy = self.data_argument_noisy(stu_x, masks=stu_masks)

        stu_x = self.stu_proj(stu_x)
        stu_x_aug_mut = self.stu_proj(stu_x_aug_mut)
        stu_x_aug_noisy = self.stu_proj(stu_x_aug_noisy)

        z1 = self.fusion_encoder([seq_x], key_padding_mask=[seq_masks])
        z2 = self.fusion_encoder([seq_x_aug_mut, stu_x_aug_mut], key_padding_mask=[seq_masks, stu_masks])
        z3 = self.fusion_encoder([seq_x_aug_noisy, stu_x_aug_noisy], key_padding_mask=[seq_masks, stu_masks])
        z4 = self.fusion_encoder([stu_x], key_padding_mask=[stu_masks])

        return {
            "encoder_out": {
                "z1": z1.pooler_output,
                "z2": z2.pooler_output,
                "z3": z3.pooler_output,
                "z4": z4.pooler_output
            },
            "encoder_states": []
        }


@register_model("fla_align_model")
class stu_seq_model(FairseqEncoderModel):

    # def __init__(self, encoder):
    #     super().__init__()
    #     self.encoder = encoder

    @staticmethod
    def add_args(parser):
        parser.add_argument('--stu_model_dim', type=int, default=1280)
        parser.add_argument('--seq_model_dim', type=int, default=1280)

        parser.add_argument('--hidden_size', type=int, default=1024)
        parser.add_argument('--num_hidden_layers', type=int, default=12)
        parser.add_argument('--num_attention_heads', type=int, default=8)
        parser.add_argument('--intermediate_size', type=int, default=4096)
        parser.add_argument('--hidden_act', type=str, default="gelu")

        parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
        parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)

        parser.add_argument('--type_vocab_size', type=int, default=2)
        parser.add_argument('--initializer_range', type=float, default=0.02)
        parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
        parser.add_argument('--pad_token_id', type=int, default=0)

        parser.add_argument('--window_size', nargs=2, type=int, default=(-1, -1),
                            help='(win_h, win_w); -1')

        parser.add_argument('--dense_seq_output', action='store_true', default=False)
        parser.add_argument('--mlp_type', type=str, default='mlp')
        parser.add_argument('--last_layer_subset', action='store_true', default=False)
        parser.add_argument('--fused_dropout_add_ln', action='store_true', default=False)
        parser.add_argument('--fused_bias_fc', action='store_true', default=False)

        parser.add_argument('--use_flash_attn', type=int, default=1)
        parser.add_argument('--use_qk_norm', type=int, default=1)


    @classmethod
    def build_model(cls, args, task):
        encoder = seq_stu_align_encoder(
            args=args, dictionary=task.target_dictionary
        )
        return cls(encoder)

    def forward(self, seq_seqs, seq_mask, stu_seqs, stu_mask, **kwargs):
        enc = self.encoder(seq_seqs, seq_mask, stu_seqs, stu_mask)
        return enc

    def max_positions(self): return 4096


@register_model_architecture("fla_align_model", "fla_align_model_base")
def stu_seq_model_base(args):
    args.stu_model_dim = getattr(args, "stu_model_dim", 1280)
    args.seq_model_dim = getattr(args, "seq_model_dim", 1280)

    args.hidden_size = getattr(args, "hidden_size", 1024)
    args.num_hidden_layers = getattr(args, "num_hidden_layers", 12)
    args.num_attention_heads = getattr(args, "num_attention_heads", 8)
    args.intermediate_size = getattr(args, "intermediate_size", 4096)
    args.hidden_act = getattr(args, "hidden_act", "gelu")

    args.hidden_dropout_prob = getattr(args, "hidden_dropout_prob", 0.1)
    args.attention_probs_dropout_prob = getattr(args, "attention_probs_dropout_prob", 0.1)

    args.initializer_range = getattr(args, "initializer_range", 0.02)
    args.layer_norm_eps = getattr(args, "layer_norm_eps", 1e-12)
    args.window_size = getattr(args, "window_size", (-1, -1))

    args.dense_seq_output = getattr(args, "dense_seq_output", False)
    args.mlp_type = getattr(args, "mlp_type", "mlp")
    args.last_layer_subset = getattr(args, "last_layer_subset", False)
    args.fused_dropout_add_ln = getattr(args, "fused_dropout_add_ln", False)
    args.fused_bias_fc = getattr(args, "fused_bias_fc", False)

    args.use_flash_attn = getattr(args, "use_flash_attn", True)
    args.use_qk_norm = getattr(args, "use_qk_norm", True)
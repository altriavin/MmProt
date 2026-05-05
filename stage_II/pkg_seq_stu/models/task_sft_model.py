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

class AAExpander(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bb_dim = args.hidden_size
        self.head_dim = args.head_dim

        self.aa_expander = nn.Linear(self.bb_dim, 20 * self.head_dim)
        self.aa_embed = nn.Embedding(20, self.head_dim)

    def forward(self, fusion_z):
        B, L, _ = fusion_z.shape
        mut1_feat = self.aa_expander(fusion_z).view(B, L, 20, self.head_dim)

        aa_embed = rearrange(self.aa_embed.weight, 'a d -> 1 1 a d')
        mut1_feat = mut1_feat + aa_embed
        return mut1_feat


class seq_stu_task_encoder(FairseqEncoder):
    def __init__(
            self, args, dictionary
    ):
        super().__init__(dictionary)
        self.model = seq_stu_align_encoder(args, dictionary)

        # pretrained_model_path = os.path.join(pretrain_root_path, args.checkpoint , "checkpoint_last.pt")
        # model_checkpoint = torch.load(pretrained_model_path, map_location='cpu')["model"]

        # enc_stripped = {k[len("encoder."):]: v for k, v in model_checkpoint.items()}

        # self.model.load_state_dict(enc_stripped, strict=True)
        # print(f'loading model from: {pretrained_model_path}')
        print(f'donot load pretraining model!!!')

        self.pred_head = nn.Linear(args.hidden_size, 33)

        # return {
        #     "id": ids,
        #     "nsentences": len(samples),
        #     "net_input": {
        #         "inputs": {
        #             "src_inputs": src_inputs,
        #             "stu_inputs": stu_inputs,
        #         }
        #     },
        #     "target": {
        #         "tgt_inputs": tgt_inputs,
        #         "mlm_masks": mlm_mask
        #     }
        # }

    def forward(self, inputs, **kwargs):
        src_inputs, stu_inputs = inputs["src_inputs"], inputs["stu_inputs"]

        mlm_fusion = self.model.forward_inference(src_inputs, stu_inputs)
        mlm_z = mlm_fusion["encoder_out"]["z"]
        mlm_probs = self.pred_head(mlm_z)

        z = mlm_fusion["encoder_out"]["z_pool"]

        return {
            "encoder_out": {
                "mlm_probs": mlm_probs,
                "z": z
            },
            "encoder_states": []
        }


@register_model("fla_inp_model")
class stu_seq_model(FairseqEncoderModel):

    # def __init__(self, encoder):
    #     super().__init__()
    #     self.encoder = encoder

    @staticmethod
    def add_args(parser):
        parser.add_argument('--stu_model_dim', type=int, default=1280)
        parser.add_argument('--seq_model_dim', type=int, default=1280)

        parser.add_argument('--hidden_size', type=int, default=1024)
        parser.add_argument('--head_dim', type=int, default=768)
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

        parser.add_argument('--checkpoint', type=str, default='checkpoint_fla_8_12')


    @classmethod
    def build_model(cls, args, task):
        encoder = seq_stu_task_encoder(
            args=args, dictionary=task.target_dictionary
        )
        return cls(encoder)

    def forward(self, inputs, **kwargs):
        enc = self.encoder(inputs)
        return enc

    def max_positions(self): return 4096


@register_model_architecture("fla_inp_model", "fla_inp_model_base")
def stu_seq_model_base(args):
    args.stu_model_dim = getattr(args, "stu_model_dim", 1280)
    args.seq_model_dim = getattr(args, "seq_model_dim", 1280)

    args.hidden_size = getattr(args, "hidden_size", 1024)
    args.head_dim = getattr(args, "head_dim", 768)

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

    args.checkpoint = getattr(args, "checkpoint", True)
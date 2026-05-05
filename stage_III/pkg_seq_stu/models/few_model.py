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
from .task_sft_model import viral_model

from einops import rearrange


class seq_stu_task_encoder(FairseqEncoder):
    def __init__(
            self, args, dictionary
    ):
        super().__init__(dictionary)
        self.viral_model = viral_model(args, dictionary)

        pretrained_model_path = os.path.join(pretrain_root_path , "checkpoint_last.pt")
        model_checkpoint = torch.load(pretrained_model_path, map_location='cpu')["model"]
        enc_stripped = {k[len("encoder."):]: v for k, v in model_checkpoint.items()}
        print(f'loading model from: {pretrained_model_path}...')

        self.viral_model.load_state_dict(enc_stripped, strict=True)

        # Cache for frozen encoder outputs (single wild-type)
        self._cached_stu_emb = None   # SaProt embedding [1, L_stu, D]
        self._cached_stu_masks = None # SaProt attention mask [1, L_stu]

    def _get_stu_cached(self, wt_stus, bs):
        """Cache and reuse SaProt embedding since all samples share the same wild-type."""
        if self._cached_stu_emb is None:
            # First call: compute and cache for a single sequence
            single_stus = {k: v[:1].cuda() for k, v in wt_stus.items()}
            single_stus["output_hidden_states"] = True
            with torch.no_grad():
                self._cached_stu_emb = self.viral_model.model.stu_encoder.model.esm(
                    **single_stus
                )["hidden_states"][-1]  # [1, L_stu, D]
            self._cached_stu_masks = wt_stus['attention_mask'][:1].eq(0).cuda()  # [1, L_stu]

        # Expand to batch size
        stu_emb = self._cached_stu_emb.expand(bs, -1, -1)
        stu_masks = self._cached_stu_masks.expand(bs, -1)
        return stu_emb, stu_masks

    def forward(self, wt, inputs, **kwargs):
        wt_seqs = wt["seqs"]
        wt_stus = wt["stus"]
        mlm_seqs = wt["mlm_seqs"]

        b_idx = inputs["batch_idx"]
        p_idx = inputs["pos_idx"]
        wt_ids = inputs["wt_ids"]
        mt_ids = inputs["mt_ids"]
        seq_m_id = inputs["seq_mask_id"]
        bs = inputs["bs"]

        # --- Optimization 1: Cache SaProt embedding (frozen, same wild-type) ---
        stu_emb, stu_masks = self._get_stu_cached(wt_stus, bs)

        # --- Optimization 2: ESM2 forward (kept separate to avoid OOM on V100) ---
        wt_seqs["input_ids"][b_idx, p_idx] = seq_m_id  # mask mutation sites

        with torch.no_grad():
            mut_seq_emb = self.viral_model.model.seq_encoder(wt_seqs)   # [bs, L, D]
            mlm_seq_emb = self.viral_model.model.seq_encoder(mlm_seqs)  # [bs, L, D]

        seq_masks = wt_seqs['attention_mask'].eq(0).cuda()  # [bs, L]

        # --- Optimization 3: Batch Fusion encoder for mut + mlm together ---
        combined_seq_emb = torch.cat([mut_seq_emb, mlm_seq_emb], dim=0)  # [2*bs, L, D]
        combined_stu_emb = stu_emb.repeat(2, 1, 1)  # [2*bs, L_stu, D]
        combined_seq_masks = seq_masks.repeat(2, 1)  # [2*bs, L]
        combined_stu_masks = stu_masks.repeat(2, 1)  # [2*bs, L_stu]

        combined_fusion = self.viral_model.model.forward_inference_cached(
            combined_seq_emb, combined_stu_emb,
            combined_seq_masks, combined_stu_masks
        )
        combined_z = combined_fusion["encoder_out"]["z"]

        z = combined_z[:bs]       # mutation scoring
        mlm_z = combined_z[bs:]   # MLM

        # --- Mutation scoring (unchanged logic) ---
        log_probs = torch.log_softmax(self.viral_model.pred_head(z), dim=-1)
        target_logits = log_probs[b_idx, p_idx]

        val_wt = target_logits.gather(1, wt_ids.view(-1, 1)).squeeze(-1)
        val_mt = target_logits.gather(1, mt_ids.view(-1, 1)).squeeze(-1)

        diffs = val_mt - val_wt

        scores = torch.zeros(bs, device="cuda").half()
        scores.index_add_(0, b_idx, diffs)

        # --- MLM probs (unchanged logic) ---
        mlm_probs = self.viral_model.pred_head(mlm_z)

        return {
            "encoder_out": {
                "scores": scores,
                "mlm_probs": mlm_probs
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

    def forward(self, wt, inputs, **kwargs):
        enc = self.encoder(wt, inputs)
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
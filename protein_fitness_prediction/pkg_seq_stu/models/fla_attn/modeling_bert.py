""" Implementation of BERT, using ALiBi and Flash Attention

The implementation was adopted from
https://github.com/Dao-AILab/flash-attention/blob/43950dda456e095969d842fca7a73c5bfe3cecd0/flash_attn/models/bert.py
and made modifications to use ALiBi.
"""

# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

import logging
from collections.abc import Sequence
from functools import partial
from typing import Union, List, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_utils import PreTrainedModel
from .configuration_bert import JinaBertConfig
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertForPreTrainingOutput,
)
from .bert_padding import (
    index_first_axis,
    index_first_axis_residual,
    pad_input,
    unpad_input,
)

from .block import Block
from .embedding import BertEmbeddings
from .mha import MHA
from .mlp import FusedMLP, Mlp, GLUMLP

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn
except ImportError:
    layer_norm_fn = None


try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None

try:
    from tqdm.autonotebook import trange
except ImportError:
    trange = None

logger = logging.getLogger(__name__)


def create_mixer_cls(config, cross_attn=False, return_residual=False):
    use_flash_attn = config.use_flash_attn if config.use_flash_attn is not None else torch.cuda.is_available()
    use_qk_norm = config.use_qk_norm
    fused_bias_fc = config.fused_bias_fc
    window_size = config.window_size
    mixer_cls = partial(
        MHA,
        num_heads=config.num_attention_heads,
        cross_attn=cross_attn,
        dropout=config.attention_probs_dropout_prob,
        causal=False,
        fused_bias_fc=fused_bias_fc,
        use_flash_attn=use_flash_attn,
        return_residual=return_residual,
        use_alibi=True,
        window_size=window_size,
        qk_norm=use_qk_norm,
        checkpointing=False,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, return_residual=False):
    inner_dim = config.intermediate_size
    mlp_type = config.mlp_type
    assert mlp_type in ('mlp', 'fused_mlp', 'glu')
    if mlp_type == 'fused_mlp':
        assert config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"], (
            "fused_mlp only " "supports approximate gelu"
        )
    if mlp_type == 'glu':
        assert config.hidden_act in ('relu', 'gelu')
    if mlp_type == 'mlp':
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate=approximate),
            return_residual=return_residual,
        )
    elif mlp_type == 'glu':
        mlp_cls = partial(
            GLUMLP,
            hidden_features=inner_dim,
            activation=config.hidden_act,
            use_flash_attn=config.use_flash_attn,
            hidden_dropout_prob=config.hidden_dropout_prob,
            return_residual=return_residual,
        )
    elif mlp_type == 'fused_mlp':
        if FusedMLP is None:
            raise ImportError("fused_dense is not installed")
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        mlp_cls = partial(
            FusedMLP,
            hidden_features=inner_dim,
            checkpoint_lvl=mlp_checkpoint_lvl,
            return_residual=return_residual,
        )
    else:
        raise NotImplementedError
    return mlp_cls


def create_block(config, layer_idx=None):
    last_layer_subset = getattr(config, "last_layer_subset", False)
    cross_attn = last_layer_subset and layer_idx == config.num_hidden_layers - 1
    # TD [2022-12-19]: For cross attention (last layer), we actually want to return the
    # residual x_kv, not residual x. But it's annoying to change the API (and it only affects
    # one layer) so we just choose not to return residual in this case.
    return_residual = not cross_attn
    mixer_cls = create_mixer_cls(config, cross_attn, return_residual=return_residual)
    mlp_cls = create_mlp_cls(config, layer_idx, return_residual=return_residual)
    norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_eps)
    block = Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=False,
        resid_dropout1=config.hidden_dropout_prob,
        resid_dropout2=config.hidden_dropout_prob,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        return_residual=return_residual,
    )
    return block


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class BertEncoder(nn.Module):
    def __init__(self, config: JinaBertConfig):
        super().__init__()
        self.use_flash_attn = config.use_flash_attn if config.use_flash_attn is not None else torch.cuda.is_available()
        self.layers = nn.ModuleList(
            [create_block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self._grad_checkpointing = False

    @property
    def gradient_checkpointing(self):
        return self._grad_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._grad_checkpointing = value

    def forward(self, hidden_states, key_padding_mask=None, subset_mask=None):
        """If subset_mask is not None, we only want output for the subset of the sequence.
        This means that we only compute the last layer output for these tokens.
        subset_mask: (batch, seqlen), dtype=torch.bool
        """
        if key_padding_mask is None or not self.use_flash_attn:
            mixer_kwargs = (
                {"key_padding_mask": key_padding_mask.bool()} if key_padding_mask is not None else None
            )
            for layer in self.layers:
                if self._grad_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer,
                        hidden_states,
                        use_reentrant=False,
                        mixer_kwargs=mixer_kwargs
                    )
                else:
                    hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
            if subset_mask is not None:
                hidden_states = hidden_states[subset_mask]
        else:
            batch, seqlen = hidden_states.shape[:2]
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                hidden_states, key_padding_mask
            )
            # print(f'indices: {indices}\ncu_seqlens: {cu_seqlens}\nmax_seqlen_in_batch: {max_seqlen_in_batch}')
            mixer_kwargs = {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen_in_batch}
            if subset_mask is None:
                for layer in self.layers:
                    if self._grad_checkpointing:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            layer,
                            hidden_states,
                            use_reentrant=False,
                            mixer_kwargs=mixer_kwargs
                        )
                    else:
                        hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                hidden_states = pad_input(hidden_states, indices, batch, seqlen)
            else:
                for layer in self.layers[:-1]:
                    if self._grad_checkpointing:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            layer,
                            hidden_states,
                            use_reentrant=False,
                            mixer_kwargs=mixer_kwargs
                        )
                    else:
                        hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                if key_padding_mask is not None:
                    subset_idx = torch.nonzero(
                        subset_mask[key_padding_mask], as_tuple=False
                    ).flatten()
                    subset_seqlens = (subset_mask & key_padding_mask).sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32), (1, 0)
                    )
                else:
                    subset_idx = torch.nonzero(subset_mask, as_tuple=False).flatten()
                    subset_seqlens = subset_mask.sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32), (1, 0)
                    )
                hidden_states_subset, hidden_states = index_first_axis_residual(
                    hidden_states, subset_idx
                )
                # It's ok to set max_seqlen_q to be much larger
                mixer_kwargs = {
                    "x_kv": hidden_states,
                    "cu_seqlens": subset_cu_seqlens,
                    "max_seqlen": max_seqlen_in_batch,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_k": max_seqlen_in_batch,
                }
                if self._grad_checkpointing:
                    torch.utils.checkpoint.checkpoint(
                        self.layers[-1],
                        hidden_states_subset,
                        use_reentrant=False,
                        mixer_kwargs=mixer_kwargs
                    )
                else:
                    hidden_states = self.layers[-1](hidden_states_subset, mixer_kwargs=mixer_kwargs)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm_fn is None:
            raise ImportError("Triton is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        self.transform_act_fn = nn.GELU(approximate=approximate)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        if not self.fused_dropout_add_ln:
            hidden_states = self.layer_norm(hidden_states)
        else:
            hidden_states = layer_norm_fn(
                hidden_states, self.layer_norm.weight, self.layer_norm.bias, eps=self.layer_norm.eps
            )
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense

        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = linear_cls(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """
    config_class = JinaBertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value


class BertModel(nn.Module):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__()
        
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm_fn is None:
            raise ImportError("Triton is not installed")

        assert config.hidden_act in ["gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"]

        self.emb_drop = nn.Dropout(config.hidden_dropout_prob)
        self.emb_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.apply(partial(_init_weights, initializer_range=config.initializer_range))

    def forward(
        self,
        x,
        key_padding_mask=None,
        return_dict=True,
    ):
        """If masked_tokens_mask is not None (i.e. last_layer_subset == True in BertForPreTraining),
        we only want the output for the masked tokens. This means that we only compute the last
        layer output for these tokens.
        masked_tokens_mask: (batch, seqlen), dtype=torch.bool
        """
        hidden_states = torch.cat(x, dim=1)
        attention_mask = ~torch.cat(key_padding_mask, dim=1)
        
        if not self.fused_dropout_add_ln:
            hidden_states = self.emb_ln(hidden_states)
        else:
            hidden_states = layer_norm_fn(
                hidden_states, self.emb_ln.weight, self.emb_ln.bias, eps=self.emb_ln.eps
            )
        hidden_states = self.emb_drop(hidden_states)

        sequence_output = self.encoder(
            hidden_states, key_padding_mask=attention_mask
        )

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output)
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


    @torch.inference_mode()
    def encode(
        self: 'BertModel',
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        output_value: str = 'sentence_embedding',
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[torch.device] = None,
        normalize_embeddings: bool = False,
        **tokenizer_kwargs,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes sentence embeddings
        Args:
            sentences(`str` or `List[str]`):
                Sentence or sentences to be encoded
            batch_size(`int`, *optional*, defaults to 32):
                Batch size for the computation
            show_progress_bar(`bool`, *optional*, defaults to None):
                Show a progress bar when encoding sentences.
                If set to None, progress bar is only shown when `logger.level == logging.INFO` or `logger.level == logging.DEBUG`.
            output_value(`str`, *optional*, defaults to 'sentence_embedding'):
                Default sentence_embedding, to get sentence embeddings.
                Can be set to token_embeddings to get wordpiece token embeddings.
                Set to None, to get all output values
            convert_to_numpy(`bool`, *optional*, defaults to True):
                If true, the output is a list of numpy vectors.
                Else, it is a list of pytorch tensors.
            convert_to_tensor(`bool`, *optional*, defaults to False):
                If true, you get one large tensor as return.
                Overwrites any setting from convert_to_numpy
            device(`torch.device`, *optional*, defaults to None):
                Which torch.device to use for the computation
            normalize_embeddings(`bool`, *optional*, defaults to False):
                If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
            tokenizer_kwargs(`Dict[str, Any]`, *optional*, defaults to {}):
                Keyword arguments for the tokenizer
        Returns:
            By default, a list of tensors is returned.
            If convert_to_tensor, a stacked tensor is returned.
            If convert_to_numpy, a numpy matrix is returned.
        """
        if self.emb_pooler is None:
            warnings.warn("No emb_pooler specified, defaulting to mean pooling.")
            self.emb_pooler = 'mean'
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self._name_or_path, trust_remote_code=True)
        if self.emb_pooler != 'mean':
            raise NotImplementedError

        is_training = self.training
        self.eval()

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True

        if device is not None:
            self.to(device)

        # TODO: Maybe use better length heuristic?
        permutation = np.argsort([-len(i) for i in sentences])
        inverse_permutation = np.argsort(permutation)
        sentences = [sentences[idx] for idx in permutation]

        tokenizer_kwargs['padding'] = tokenizer_kwargs.get('padding', True)
        tokenizer_kwargs['max_length'] = tokenizer_kwargs.get('max_length', 8192)
        tokenizer_kwargs['truncation'] = tokenizer_kwargs.get('truncation', True)

        all_embeddings = []

        if trange is not None:
            range_iter = trange(
                0,
                len(sentences),
                batch_size,
                desc="Encoding",
                disable=not show_progress_bar,
            )
        else:
            range_iter = range(0, len(sentences), batch_size)

        for i in range_iter:
            encoded_input = self.tokenizer(
                sentences[i : i + batch_size],
                return_tensors='pt',
                **tokenizer_kwargs,
            ).to(self.device)
            token_embs = self.forward(**encoded_input)[0]

            # Accumulate in fp32 to avoid overflow
            token_embs = token_embs.float()

            if output_value == 'token_embeddings':
                raise NotImplementedError
            elif output_value is None:
                raise NotImplementedError
            else:
                embeddings = self.mean_pooling(
                    token_embs, encoded_input['attention_mask']
                )

                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                if convert_to_numpy:
                    embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in inverse_permutation]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        self.train(is_training)
        return all_embeddings

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config: JinaBertConfig):
        super().__init__(config)
        # If dense_seq_output, we only need to pass the hidden states for the masked out tokens
        # (around 15%) to the classifier heads.
        self.dense_seq_output = getattr(config, "dense_seq_output", False)
        # If last_layer_subset, we only need the compute the last layer for a subset of tokens
        # (e.g., the tokens we need to compute the masked LM loss and the next-sentence prediction).
        self.last_layer_subset = getattr(config, "last_layer_subset", False)
        if self.last_layer_subset:
            assert self.dense_seq_output, "last_layer_subset requires dense_seq_output"
        use_xentropy = getattr(config, "use_xentropy", False)
        if use_xentropy and CrossEntropyLoss is None:
            raise ImportError("xentropy_cuda is not installed")
        loss_cls = (
            nn.CrossEntropyLoss
            if not use_xentropy
            else partial(CrossEntropyLoss, inplace_backward=True)
        )

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.mlm_loss = loss_cls(ignore_index=0)
        self.nsp_loss = loss_cls(ignore_index=-1)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        next_sentence_label=None,
    ):
        """
        If labels are provided, they must be 0 for masked out tokens (as specified in the attention
        mask).
        Outputs:
            if `labels` and `next_sentence_label` are not `None`:
                Outputs the total_loss which is the sum of the masked language modeling loss and the next
                sentence classification loss.
            if `labels` or `next_sentence_label` is `None`:
                Outputs a tuple comprising
                - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
                - the next sentence classification logits of shape [batch_size, 2].

        """
        masked_tokens_mask = labels > 0 if (self.last_layer_subset and labels is not None) else None
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            masked_tokens_mask=masked_tokens_mask,
        )
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output
        if self.dense_seq_output and labels is not None:
            masked_token_idx = torch.nonzero(labels.flatten() > 0, as_tuple=False).flatten()
            if not self.last_layer_subset:
                sequence_output = index_first_axis(
                    rearrange(sequence_output, "b s d -> (b s) d"), masked_token_idx
                )
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if (
            self.dense_seq_output and labels is not None
        ):  # prediction_scores are already flattened
            masked_lm_loss = self.mlm_loss(
                prediction_scores, labels.flatten()[masked_token_idx]
            ).float()
        elif labels is not None:
            masked_lm_loss = self.mlm_loss(
                rearrange(prediction_scores, "... v -> (...) v"),
                rearrange(labels, "... -> (...)"),
            ).float()
        else:
            masked_lm_loss = 0
        if next_sentence_label is not None:
            next_sentence_loss = self.nsp_loss(
                rearrange(seq_relationship_score, "... t -> (...) t"),
                rearrange(next_sentence_label, "... -> (...)"),
            ).float()
        else:
            next_sentence_loss = 0

        total_loss = masked_lm_loss + next_sentence_loss

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
        )


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config: JinaBertConfig):
        super().__init__(config)
        # If dense_seq_output, we only need to pass the hidden states for the masked out tokens
        # (around 15%) to the classifier heads.
        self.dense_seq_output = getattr(config, "dense_seq_output", False)
        # If last_layer_subset, we only need the compute the last layer for a subset of tokens
        # (e.g., the tokens we need to compute the masked LM loss and the next-sentence prediction).
        self.last_layer_subset = getattr(config, "last_layer_subset", False)
        if self.last_layer_subset:
            assert self.dense_seq_output, "last_layer_subset requires dense_seq_output"
        use_xentropy = getattr(config, "use_xentropy", False)
        if use_xentropy and CrossEntropyLoss is None:
            raise ImportError("xentropy_cuda is not installed")
        loss_cls = (
            nn.CrossEntropyLoss
            if not use_xentropy
            else partial(CrossEntropyLoss, inplace_backward=True)
        )

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.mlm_loss = loss_cls(ignore_index=0)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None
    ):
        masked_tokens_mask = labels > 0 if (self.last_layer_subset and labels is not None) else None
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            masked_tokens_mask=masked_tokens_mask,
        )
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output
        if self.dense_seq_output and labels is not None:
            masked_token_idx = torch.nonzero(labels.flatten() > 0, as_tuple=False).flatten()
            if not self.last_layer_subset:
                sequence_output = index_first_axis(
                    rearrange(sequence_output, "b s d -> (b s) d"), masked_token_idx
                )
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if (
                self.dense_seq_output and labels is not None
        ):  # prediction_scores are already flattened
            masked_lm_loss = self.mlm_loss(
                prediction_scores, labels.flatten()[masked_token_idx]
            ).float()
        elif labels is not None:
            masked_lm_loss = self.mlm_loss(
                rearrange(prediction_scores, "... v -> (...) v"),
                rearrange(labels, "... -> (...)"),
            ).float()
        else:
            raise ValueError('MLM labels must not be None')

        return BertForPreTrainingOutput(
            loss=masked_lm_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
        )

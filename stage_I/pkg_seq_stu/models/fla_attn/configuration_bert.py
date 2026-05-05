# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT model configuration"""

from transformers import PretrainedConfig


class JinaBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        window_size (`tuple`, *optional*, defaults to `(-1, -1)`): If not the default, use local attention
    """

    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        window_size=(-1, -1),
        dense_seq_output=False,
        mlp_type='mlp',
        mlp_checkpoint_lvl=0,
        last_layer_subset=False,
        fused_dropout_add_ln=False,
        fused_bias_fc=False,
        pad_vocab_size_multiple=1,
        use_flash_attn=True,
        use_qk_norm=True,
        emb_pooler=None,
        classifier_dropout=None,
        num_loras=5,
        **kwargs,
    ):
        assert 'position_embedding_type' not in kwargs
        assert 'max_position_embeddings' not in kwargs
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        if mlp_type == 'fused_mlp' and hidden_act not in ["gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"]:
            raise ValueError('Fused MLP only supports approximate gelu')

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.window_size = window_size
        self.dense_seq_output = dense_seq_output
        self.mlp_type= mlp_type
        self.mlp_checkpoint_lvl = mlp_checkpoint_lvl
        self.last_layer_subset = last_layer_subset
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.fused_bias_fc = fused_bias_fc
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.use_flash_attn = use_flash_attn
        self.use_qk_norm = use_qk_norm
        self.emb_pooler = emb_pooler
        self.classifier_dropout = classifier_dropout
        self.num_loras = num_loras
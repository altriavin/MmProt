# Copyright (c) 2023, Tri Dao.

""""
The implementation was adopted from
https://github.com/Dao-AILab/flash-attention/blob/43950dda456e095969d842fca7a73c5bfe3cecd0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup


try:
    from flash_attn.ops.activations import swiglu
except ImportError:
    swiglu = None

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
except ImportError:
    ColumnParallelLinear, RowParallelLinear = None, None

try:
    from flash_attn.ops.fused_dense import FusedMLP, ParallelFusedMLP
except ImportError:
    FusedMLP, ParallelFusedMLP = None, None


class GLUMLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features,
            activation,
            use_flash_attn,
            return_residual=False,
            hidden_dropout_prob=0.1
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.gated_layers = nn.Linear(
            in_features, hidden_features * 2, bias=False
        )
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(
                f"activation {activation} not supported"
            )
        self.wo = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.return_residual = return_residual
        self.use_flash_attn = use_flash_attn
        #self.layernorm = nn.LayerNorm(in_features, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        if self.use_flash_attn:
            gated = hidden_states[:, : self.hidden_features]
            non_gated = hidden_states[:, self.hidden_features :]
        else:
            gated = hidden_states[:, :, : self.hidden_features]
            non_gated = hidden_states[:, :, self.hidden_features :]
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        # hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states if not self.return_residual else (hidden_states, residual_connection)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        bias1=True,
        bias2=True,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features * 4
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class ParallelMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        process_group: ProcessGroup = None,
        sequence_parallel=True,
        bias1=True,
        bias2=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert ColumnParallelLinear is not None, "Need to install fused_dense"
        assert RowParallelLinear is not None, "Need to install fused_dense"
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features * 4
        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            process_group,
            bias=bias1,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.activation = activation
        self.fc2 = RowParallelLinear(
            hidden_features,
            out_features,
            process_group,
            bias=bias2,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y


class GatedMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.sigmoid,
        bias1=True,
        bias2=True,
        multiple_of=128,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias1, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        if self.activation == F.sigmoid:  # Special case for GLU
            y = F.glu(y, dim=-1)
        elif self.activation == F.silu and swiglu is not None:  # Special case for SwiGLU
            y, gate = y.chunk(2, dim=-1)
            y = swiglu(gate, y)
        else:
            y, gate = y.chunk(2, dim=-1)
            y = y * self.activation(gate)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class ParallelGatedMlp(nn.Module):
    """Parallel GatedMlp"""

    def __init__(
        self,
        in_features,
        process_group,
        hidden_features=None,
        out_features=None,
        activation=F.sigmoid,
        bias1=True,
        bias2=True,
        multiple_of=128,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.fc1 = ColumnParallelLinear(
            in_features,
            2 * hidden_features,
            process_group,
            bias=bias1,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.activation = activation
        self.fc2 = RowParallelLinear(
            hidden_features,
            out_features,
            process_group,
            bias=bias2,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )

    def forward(self, x):
        y = self.fc1(x)
        if self.activation == F.sigmoid:  # Special case for GLU
            y = F.glu(y, dim=-1)
        else:
            y, gate = y.chunk(2, dim=-1)
            y = y * self.activation(gate)
        y = self.fc2(y)
        return y

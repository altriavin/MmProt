import math
import os
from functools import partial
from typing import Iterator, Optional, Tuple, Union

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn
from torch.nn import Parameter
from transformers import PretrainedConfig

from .modeling_bert import BertModel, BertPreTrainedModel, JinaBertConfig


def initialized_weights(
    shape: Tuple[int], num_adaptions: int, init: str = "kaiming"
) -> torch.Tensor:
    weight_data = []
    for _ in range(num_adaptions):
        new_adaption = torch.zeros(shape)
        if init == "kaiming":
            nn.init.kaiming_uniform_(new_adaption, a=math.sqrt(5))
        elif init == "normal":
            nn.init.normal_(new_adaption)
        else:
            raise NotImplementedError
        weight_data.append(new_adaption)
    return torch.stack(weight_data, dim=0)


class LoRAParametrization(nn.Module):
    """
    This LoRA implementation was inspired by  https://github.com/cccntu/minLoRA

    The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: str = "linear",
        num_adaptions: int = 1,
        rank: int = 4,
        lora_dropout_p: float = 0.0,
        lora_alpha: float = 1,
    ):
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        fan_in_fan_out = layer_type == "embedding"
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)

        # For the officially "correct" LoRA initialization, check here: https://github.com/microsoft/LoRA
        # TODO: Ensure that the initialization here is correct
        if layer_type == "linear":
            self.lora_A = nn.Parameter(
                initialized_weights((rank, fan_in), num_adaptions, init="kaiming")
            )
            self.lora_B = nn.Parameter(torch.zeros((num_adaptions, fan_out, rank)))
        elif layer_type == "embedding":
            self.lora_A = nn.Parameter(torch.zeros((num_adaptions, fan_in, rank)))
            self.lora_B = nn.Parameter(
                initialized_weights(
                    (rank, fan_out), num_adaptions=num_adaptions, init="normal"
                )
            )
        else:
            raise NotImplementedError

        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = (
            nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        )
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer(
            "lora_dropout_mask",
            torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype),
            persistent=False,
        )
        self.forward_fn = lambda x: x
        self.current_task = None

    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)

    def lora_forward(self, X):
        assert self.current_task is not None
        return (
            X
            + torch.matmul(
                *self.swap(
                    (
                        self.lora_B[self.current_task],
                        self.dropout_fn(self.lora_A[self.current_task]),
                    )
                )
            ).view(X.shape)
            * self.scaling
        )

    def forward(self, X):
        return self.forward_fn(X)

    @property
    def current_task(self):
        return self._current_task

    @current_task.setter
    def current_task(self, task: Union[None, int]):
        self._current_task = task
        if task is None:
            self.forward_fn = lambda x: x
        else:
            self.forward_fn = self.lora_forward

    @classmethod
    def from_linear(
        cls,
        layer: nn.Module,
        num_adaptions: int = 1,
        rank: int = 4,
        lora_dropout_p: float = 0.0,
        lora_alpha: int = 1,
    ):
        assert isinstance(layer, nn.Linear)
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            num_adaptions=num_adaptions,
            layer_type="linear",
            rank=rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
        )

    @classmethod
    def from_embedding(
        cls, layer, num_adaptions=1, rank=4, lora_dropout_p=0.0, lora_alpha=1
    ):
        assert isinstance(layer, nn.Embedding)
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            num_adaptions=num_adaptions,
            layer_type="embedding",
            rank=rank,
            lora_dropout_p=lora_dropout_p,
            lora_alpha=lora_alpha,
        )

    @classmethod
    def add_to_layer(
        cls, layer, num_adaptions=1, rank=4, lora_dropout_p=0.0, lora_alpha=1
    ):
        if isinstance(layer, nn.Linear):
            parametrize.register_parametrization(
                layer,
                "weight",
                cls.from_linear(
                    layer,
                    num_adaptions=num_adaptions,
                    rank=rank,
                    lora_dropout_p=lora_dropout_p,
                    lora_alpha=lora_alpha,
                ),
            )
        elif isinstance(layer, nn.Embedding):
            parametrize.register_parametrization(
                layer,
                "weight",
                cls.from_embedding(
                    layer,
                    num_adaptions=num_adaptions,
                    rank=rank,
                    lora_dropout_p=lora_dropout_p,
                    lora_alpha=lora_alpha,
                ),
            )

    @staticmethod
    def select_task_for_layer(layer: nn.Module, task_idx: Optional[int] = None):
        if isinstance(layer, LoRAParametrization):
            layer.current_task = task_idx

    @staticmethod
    def merge_lora_into_layer(layer: nn.Module):
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=True)


class BertLoRA(BertPreTrainedModel):
    def __init__(self, config: JinaBertConfig, bert: Optional[BertModel] = None, add_pooling_layer=True):
        super().__init__(config)
        if bert is None:
            self.bert = BertModel(config, add_pooling_layer=add_pooling_layer)
        else:
            self.bert = bert
        self._is_merged = False
        self._num_adaptions = config.num_loras
        self._register_lora(self._num_adaptions)
        self.main_params_trainable = False
        self._task_idx = None
        # By default, we select the first LoRA
        self.current_task = 0

    @property
    def main_params_trainable(self):
        return self._main_params_trainable

    @main_params_trainable.setter
    def main_params_trainable(self, val: bool):
        """Whether the main parameters (i.e. those that are not LoRA) should be trainable.

        This method sets the `requires_grad_` attribute of the main weights
        and controls which parameters are returned in `self.parameters()`.

        :param val: Whether or not to make the parameters trainable.
        :return: None
        """
        self._main_params_trainable = val
        for name, param in super().named_parameters():
            if "lora" not in name:
                param.requires_grad_(val)

    @classmethod
    def from_bert(cls, *args, **kwargs):
        bert = BertModel.from_pretrained(*args, **kwargs)
        config = JinaBertConfig.from_pretrained(*args, **kwargs)
        return cls(config, bert=bert)

    def merge_lora(self):
        """Merges currently selected LoRA into main weights."""
        if self._is_merged:
            raise Exception('LoRA has already been merged, cannot merge again')
        self._is_merged = True
        self.apply(LoRAParametrization.merge_lora_into_layer)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        """
        TODO: choose between from_bert and super().from_pretrained

        We want to be able to load both a pretrained BertModel, and a trained
        BertLoRA via this method. To this end, we need to check which of these
        models we are expected to load.
        """
        return cls.from_bert(pretrained_model_name_or_path)

    def _register_lora(self, num_adaptions=1, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        self.apply(
            partial(
                LoRAParametrization.add_to_layer,
                num_adaptions=num_adaptions,
                rank=rank,
                lora_dropout_p=lora_dropout_p,
                lora_alpha=lora_alpha,
            )
        )

    @property
    def current_task(self):
        """ Which LoRA is currently selected
        :return: Integer or None (when LoRA is disabled)
        """
        return self._task_idx

    @current_task.setter
    def current_task(self, task_idx: Union[None, int]):
        """Set the LoRA that is to be used.

        The LoRA is specified by `task_idx`, which may be an integer >= 0,
        indexing the available LoRAs. If it is None, no LoRA is used.

        :param task_idx: Which LoRA to use
        :return:
        """
        if self._is_merged:
            raise Exception('LoRA has been merged, cannot select new task')
        assert task_idx is None or 0 <= task_idx < self._num_adaptions
        if self._task_idx != task_idx:
            # In this case, we need to update the LoRAs everywhere
            self._task_idx = task_idx
            self.apply(
                partial(LoRAParametrization.select_task_for_layer, task_idx=task_idx)
            )

    def forward(self, *args, current_task: Union[None, int] = -1, **kwargs):
        if current_task is None or current_task >= 0:
            self.current_task = current_task
        return self.bert(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        for name, param in super().named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        ):
            if "lora" in name or self.main_params_trainable:
                yield name, param

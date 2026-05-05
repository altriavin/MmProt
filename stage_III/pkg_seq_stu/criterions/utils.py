from typing import List, Sequence, Union
import torch
import torch.distributed as dist

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import numpy as np


class _AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int = 0, group=None):
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size(group) == 1:
            ctx.world_size = 1
            ctx.rank = 0
            ctx.dim = dim
            ctx.sizes = torch.tensor([x.size(dim)], device=x.device, dtype=torch.long)
            return x

        ws = dist.get_world_size(group)
        rk = dist.get_rank(group)
        ctx.world_size = ws
        ctx.rank = rk
        ctx.dim = dim
        device = x.device

        local_len = torch.tensor([x.size(dim)], device=device, dtype=torch.long)
        sizes_list = [torch.zeros_like(local_len) for _ in range(ws)]
        dist.all_gather(sizes_list, local_len, group=group)
        sizes = torch.stack(sizes_list).squeeze(-1)
        max_len = int(sizes.max().item())
        ctx.sizes = sizes
        ctx.max_len = max_len

        if max_len > x.size(dim):
            pad_shape = list(x.shape)
            pad_shape[dim] = max_len - x.size(dim)
            x = torch.cat([x, x.new_zeros(pad_shape)], dim=dim)

        gathered = [torch.empty_like(x) for _ in range(ws)]
        dist.all_gather(gathered, x, group=group)
        big = torch.cat(gathered, dim=dim)

        chunks = []
        for i in range(ws):
            si = int(sizes[i].item())
            if si == 0:
                continue
            start = i * max_len
            chunks.append(big.narrow(dim, start, si))
        out = torch.cat(chunks, dim=dim)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        dim = ctx.dim
        ws = ctx.world_size
        if ws == 1:
            return grad_out, None, None

        sizes = ctx.sizes.tolist()
        splits = sizes
        grads = torch.split(grad_out, splits, dim=dim)
        grad_local = grads[ctx.rank].contiguous()
        return grad_local, None, None


def _all_gather_one_with_grad(x: torch.Tensor, dim: int = 0, group=None) -> torch.Tensor:
    return _AllGatherWithGrad.apply(x, dim, group)


def all_gather_batch_with_grad(
    tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
    dim: int = 0,
    group=None
):
    if isinstance(tensors, torch.Tensor):
        return _all_gather_one_with_grad(tensors, dim=dim, group=group)
    elif isinstance(tensors, (list, tuple)):
        gathered = [ _all_gather_one_with_grad(t, dim=dim, group=group) for t in tensors ]
        return tuple(gathered) if isinstance(tensors, tuple) else gathered
    else:
        raise TypeError(f"Unsupported type {type(tensors)} for all_gather_batch_with_grad")


def eval_reg(targets, preds):
    targets = targets.reshape(-1)
    preds = preds.reshape(-1)

    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)

    if len(targets) < 2:
        return {'RMSE': rmse, 'PCC': float('nan'), 'SCC': float('nan')}

    pcc, p_value = pearsonr(targets, preds)
    scc, s_p_value = spearmanr(targets, preds)

    return {
        'RMSE': rmse,
        'PCC': pcc,
        'SCC': scc
    }
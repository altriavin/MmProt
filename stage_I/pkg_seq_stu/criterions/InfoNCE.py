import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import all_gather_batch_with_grad

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, default_gather=True):
        super().__init__()
        self.temperature = float(temperature)
        self.default_gather = default_gather
        self.INF = 1e8


    def forward(self, z1, z2, gather=None):
        if gather is None:
            gather = self.default_gather

        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        if gather:
            z1, z2 = all_gather_batch_with_grad([z1, z2])

        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = F.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()

        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N

        # with torch.no_grad():
        #     diag_pos = (z1 * z2).sum(-1) / self.temperature
        #     row_max_neg = torch.topk(((z1 @ z2.t()) / self.temperature).fill_diagonal_(-1e9), k=1, dim=1).values.squeeze(-1)
        #     assert (diag_pos > row_max_neg).float().mean() > 0.5, "align error!!!!"

        return loss

    def forward_bak(self, z1, z2, gather=None):
        if gather is None:
            gather = self.default_gather

        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        if gather:
            z1, z2 = all_gather_batch_with_grad([z1, z2])

        logits = (z1 @ z2.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a + loss_b)

    def __str__(self):
        return f"InfoNCE(temp={self.temperature}, default_gather={self.default_gather})"
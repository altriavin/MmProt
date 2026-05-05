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
    

class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            features = F.normalize(features, p=2, dim=-1)
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)



class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, wt_emb, mt_emb, label_wt, label_mt):
        features = torch.cat([wt_emb, mt_emb], dim=0)  # [2bs, feat_dim]
        labels = torch.cat([label_wt, label_mt], dim=0).unsqueeze(1)

        # print(f'features.shape: {features.shape}; labels.shape: {labels.shape}')

        label_diffs = self.label_diff_fn(labels)
        # print(f'label_diffs: {label_diffs}')

        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs


        logits = logits.masked_select((1 - torch.eye(n).cuda()).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).cuda()).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).cuda()).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss


def BT_loss(scores, golden_score):

    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    g_diff = golden_score.unsqueeze(1) - golden_score.unsqueeze(0)
    signs = torch.where(g_diff > 0, -1.0, 1.0)
    logits = signs * s_diff
    loss_matrix = F.softplus(logits)
    mask = torch.triu(torch.ones_like(loss_matrix))
    loss = (loss_matrix * mask).sum() / mask.sum()
    return loss


def KLloss(logits, logits_reg):
    # device = logits.device

    # log_p = F.log_softmax(logits, dim=-1)
    # q = F.softmax(logits_reg, dim=-1)

    kl = F.kl_div(log_p, q, reduction='batchmean')
    # loss = kl.sum(dim=-1).mean()
    # mask = att_mask.to(device).unsqueeze(-1)
    # kl = (kl * mask).sum(dim=-1)

    # denom = mask.sum()
    # loss = kl.sum() / denom

    return kl



# def BT_loss(scores, golden_score):
#     loss = torch.tensor(0.)
#     loss = loss.cuda()
#     for i in range(len(scores)):
#         for j in range(i, len(scores)):
#             if golden_score[i] > golden_score[j]:
#                 loss += torch.log(1+torch.exp(scores[j]-scores[i]))
#             else:
#                 loss += torch.log(1+torch.exp(scores[i]-scores[j]))
#     return loss / len(scores)


class CustomMultiLossLayer(nn.Module):
    """
    Inspired by
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """

    def __init__(self, loss_num):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss


def lambdaLoss(y_pred, y_true, eps=1e-8, padded_value_indicator=-100, weighing_scheme="lambdaRank_scheme", k=None, sigma=1., mu=10.,
               reduction="sum", reduction_log="binary"):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    # print(f'y_pred: {y_pred}\ny_true: {y_true}')

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # print(f'y_pred_sorted: {y_pred_sorted}')
    # print(f't_true_sorted: {y_true_sorted}')

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    slate_len = y_pred.shape[1]
    if k is None or k > slate_len:
        k = slate_len

    ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.
    else:
        weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    # print(f'scores_diffs: {scores_diffs}')
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))


class MaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.

    Evaluates the cross-entropy loss at specified locations in a sequence.

    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - weight: (C, ): class weights for nn.CrossEntropyLoss
    """

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, pred, tgt, mask):
        # Make sure we have that empty last dimension

        # print(f'pred.shape: {pred}')
        # print(f'tgt.shape: {tgt}')
        # print(f'mask.shape: {mask}')

        if len(mask.shape) == len(pred.shape) - 1:
            mask = mask.unsqueeze(-1)
        # Make sure mask is boolean
        mask = mask.bool()
        # Number of locations to calculate loss
        n = mask.sum()
        # Select
        p = torch.masked_select(pred, mask).view(n, -1)
        t = torch.masked_select(tgt.cuda(), mask.squeeze())
        return super().forward(p, t)
import math
import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics

from .utils import eval_reg
from .InfoNCE import RnCLoss, BT_loss, CustomMultiLossLayer, KLloss, lambdaLoss, MaskedCrossEntropyLoss

from torch import nn
import torch.distributed as dist
import numpy as np

from pkg_seq_stu.models.seq_encoder import export_encoder


@register_criterion("fla_inp_criterion")
class seq_stu_criterion(FairseqCriterion):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--beta", type=float, default=2.0, help="SimPO reward scaling factor")
        parser.add_argument("--gamma", type=float, default=2.0, help="SimPO target reward margin")
        parser.add_argument("--task_loss_weight", type=float, default=1.0)

        parser.add_argument("--label-temperature", type=float, default=0.1,
                            help="Temperature for converting scalar labels to probabilities. Lower values make the target distribution sharper.")

    def __init__(self, task):
        super().__init__(task)

        self.temperature = getattr(task.args, "temperature", 1.0)
        self.beta = getattr(task.args, "beta", 2.0)
        self.gamma = getattr(task.args, "gamma", 2.0)
        self.label_temperature = getattr(task.args, "label_temperature", 0.1)

        self.task_weight = getattr(task.args, "task_loss_weight", 1.0)

        self.mlm_loss_func = MaskedCrossEntropyLoss()
        # self.rnk_loss_fn = RnCLoss(...)
        # self.expert_model = export_encoder()

    # def compute_weighted_simpo_loss(self, model_scores, labels):

    #     target_scores = labels.float()
    #     model_scores = model_scores.view(-1).float()
    #     target_scores = target_scores.view(-1)

    #     N = model_scores.size(0)
    #     device = model_scores.device

    #     S_i = target_scores.unsqueeze(1)
    #     S_j = target_scores.unsqueeze(0)

    #     diff_labels = (S_i - S_j) / self.label_temperature
    #     w_ij = torch.sigmoid(diff_labels).detach()

    #     # print(f'diff_labels: {diff_labels}')
    #     # print(f'w_ij: {w_ij}')

    #     r_i = model_scores.unsqueeze(1)
    #     r_j = model_scores.unsqueeze(0)

    #     diff_rewards = self.beta * (r_i - r_j)
    #     simpo_term = F.logsigmoid(diff_rewards - self.gamma)

    #     # print(f'simpo_term: {simpo_term}')

    #     mask = ~torch.eye(N, dtype=torch.bool, device=device)
    #     weighted_loss = - (w_ij * simpo_term)
    #     loss = weighted_loss[mask].mean()

    #     return loss

    def forward(self, model, sample):
        net_output = model.encoder(**sample["net_input"])

        mlm_probs = net_output["encoder_out"]["mlm_probs"]
        mlm_probs = mlm_probs[:, :mlm_probs.shape[1] // 2, :]
        mlm_src = sample["net_input"]["wt"]["seqs"]["input_ids"]
        mlm_masks = sample["target"]["mlm_masks"]

        mlm_loss = self.mlm_loss_func(mlm_probs, mlm_src, mlm_masks)

        logits = net_output["encoder_out"]["scores"]
        labels = sample["target"]["labels"]

        lambda_loss = BT_loss(logits, labels)

        loss = self.beta * lambda_loss  + self.gamma * mlm_loss

        loss_item = {
            "loss": loss.item(),
            "rank_loss": lambda_loss.item(),
            "mlm_loss": mlm_loss.item(),
        }

        logging_output = {**loss_item}

        if not model.training:
            logging_output["predictions"] = logits.detach().cpu()
            logging_output["targets"] = labels.detach().cpu()

        return loss, len(sample), logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0.0) for log in logging_outputs)
        n = max(1, len(logging_outputs))
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / n, sample_size, round=3)

        task_loss_sum = sum(log.get("rank_loss", 0.0) for log in logging_outputs)
        cl_loss_sum = sum(log.get("mlm_loss", 0.0) for log in logging_outputs)

        metrics.log_scalar("rank_loss",  task_loss_sum / n, sample_size, round=3)
        metrics.log_scalar("mlm_loss",  cl_loss_sum / n, sample_size, round=3)

        if len(logging_outputs) > 0 and "predictions" in logging_outputs[0]:
            pred_list = [log["predictions"] for log in logging_outputs]
            target_list = [log["targets"] for log in logging_outputs]

            all_preds = torch.cat(pred_list, dim=0)
            all_targets = torch.cat(target_list, dim=0)

            result_reg = eval_reg(all_preds.numpy(), all_targets.numpy())

            if "RMSE" in result_reg:
                metrics.log_scalar("RMSE", result_reg["RMSE"], 1, round=4)
            if "PCC" in result_reg:
                metrics.log_scalar("PCC", result_reg["PCC"], 1, round=4)
            if "SCC" in result_reg:
                metrics.log_scalar("SCC", result_reg["SCC"], 1, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
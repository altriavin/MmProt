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

    def forward(self, model, sample):
        net_output = model.encoder(**sample["net_input"])
        src_inputs = sample["net_input"]["inputs"]["src_inputs"]["input_ids"]

        mlm_probs = net_output["encoder_out"]["mlm_probs"]
        mlm_probs = mlm_probs[:, :src_inputs.shape[1], :]

        mlm_tgt = sample["target"]["tgt_inputs"]
        mlm_masks = sample["target"]["mlm_masks"]

        loss = self.mlm_loss_func(mlm_probs, mlm_tgt, mlm_masks)

        loss_item = {
            "loss": loss.item(),
        }

        logging_output = {**loss_item}

        if not model.training:
            with torch.no_grad():
                pred_tokens = mlm_probs.argmax(dim=-1)
                correct = ((pred_tokens == mlm_tgt) & mlm_masks).sum().item()
                total = int(mlm_masks.sum().item())

            logging_output.update({
                "mlm_correct": correct,
                "mlm_total": total,
            })

        return loss, len(sample), logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0.0) for log in logging_outputs)
        n = max(1, len(logging_outputs))
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / n, sample_size, round=3)

        correct = sum(log.get("mlm_correct", 0.0) for log in logging_outputs)
        total = sum(log.get("mlm_total", 0.0) for log in logging_outputs)
        if total > 0:
            metrics.log_scalar("mlm_acc", 100.0 * correct / total, total, round=2)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
import math
import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics
from .InfoNCE import InfoNCE


@register_criterion("fla_align_criterion")
class seq_stu_criterion(FairseqCriterion):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--token-loss-weight", type=float, default=1.0)
        parser.add_argument("--protein-loss-weight", type=float, default=1.0)
        parser.add_argument("--token-gather", action="store_true", default=False, help="")

    def __init__(self, task):
        super().__init__(task)
        tau = getattr(task.args, "temperature", 0.1)
        self.w_token = getattr(task.args, "token_loss_weight", 1.0)
        self.w_protein = getattr(task.args, "protein_loss_weight", 1.0)
        self.token_gather = getattr(task.args, "token_gather", False)

        self.loss_fn = InfoNCE(temperature=tau, default_gather=True)

    def forward(self, model, sample, reduce=True):
        # print(f'sample: {sample}')
        enc = model.encoder(**sample["net_input"])
        z1, z2, z3, z4 = enc["encoder_out"]["z1"], enc["encoder_out"]["z2"], enc["encoder_out"]["z3"], \
        enc["encoder_out"]["z4"]

        # print(f'z1: {z1}\nz2: {z2}\nz3: {z3}\nz4: {z4}')

        # print(f'z1.shape: {z1.shape}; z2.shape: {z2.shape}; z3.shape: {z3.shape}; z4.shape: {z4.shape}')

        loss_1 = self.loss_fn(z2, z3)
        loss_2_1 = self.loss_fn(z1, z2)
        loss_2_2 = self.loss_fn(z1, z3)
        loss_2 = (loss_2_1 + loss_2_2) / 2.0

        loss_3_1 = self.loss_fn(z4, z2)
        loss_3_2 = self.loss_fn(z4, z3)
        loss_3 = (loss_3_1 + loss_3_2) / 2.0
        
        loss = (loss_1 + loss_2 + loss_3) / 3.0

        # print(f'loss_1: {loss_1}; loss_2: {loss_2}; loss_3: {loss_3}')

        logging_output = {
            "loss": loss.item()
        }
        return loss, len(sample), logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0.0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum, len(logging_outputs))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
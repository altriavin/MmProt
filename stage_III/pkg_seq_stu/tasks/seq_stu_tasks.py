import os
from typing import List
import numpy as np
import torch
from fairseq.tasks import FairseqTask, register_task
from fairseq.data import FairseqDataset, data_utils
from transformers import EsmTokenizer, EsmForMaskedLM

import pandas as pd
from pkg_seq_stu.constants import (
    esm_root_path, saport_root_path, data_root_path, AA_TO_IDX,
    dms_csv_path, FEW_SHOT_SIZES
)
from pkg_seq_stu.criterions.utils import eval_reg
from .data_augment import mlm_seq_pred

class SSDataset(FairseqDataset):
    def __init__(self, stus, muts, scores):
        self.stus = stus
        self.muts = muts
        self.labels = scores

        self.sizes = np.fromiter((len(x) for x in self.stus), dtype=np.int64)

        self.seq_tokenizer = EsmTokenizer.from_pretrained(
            esm_root_path, weights_only=False
        )
        self.stu_tokenizer = EsmTokenizer.from_pretrained(
            saport_root_path, weights_only=False
        )

        # Pre-tokenize the single wild-type sequence (shared by all samples)
        wt_seq = self.stus[0][::2]  # extract AA sequence from structure-aware token
        wt_stu = self.stus[0]
        self._wt_seq_tokens = self.seq_tokenizer(wt_seq, return_tensors="pt")
        self._wt_stu_tokens = self.stu_tokenizer(wt_stu, return_tensors="pt")
        self._seq_m_id = self.seq_tokenizer.mask_token_id

    def __getitem__(self, index):
        return {"id": index, "stu": self.stus[index], "mut": self.muts[index], "label": self.labels[index]}

    def __len__(self):
        return len(self.stus)

    def collater(self, samples):

        if len(samples) == 0:
            return {}

        ids, seqs = [], []
        muts, labels = [], []
        for item in samples:
            ids.append(item["id"])
            seqs.append(item["stu"][::2])
            muts.append(item["mut"])
            labels.append(item["label"])

        bs = len(samples)
        mlm_seq, _, mlm_mask = mlm_seq_pred(seqs)

        # Reuse pre-tokenized wild-type tokens (expand to batch size)
        wt_seq_inputs = {k: v.expand(bs, -1).clone() for k, v in self._wt_seq_tokens.items()}

        mlm_seq_inputs = self.seq_tokenizer(
            mlm_seq,
            padding=True,
            return_tensors="pt"
        )
        mlm_seq_inputs['attention_mask'] = wt_seq_inputs["attention_mask"]

        wt_stu_inputs = {k: v.expand(bs, -1).clone() for k, v in self._wt_stu_tokens.items()}

        batch_indices = []
        pos_indices = []
        wt_token_ids = []
        mt_token_ids = []

        for i, mut_str in enumerate(muts):
            for m in mut_str.split(','):
                wt_char = m[0]
                pos = int(m[1:-1])
                mut_char = m[-1]

                batch_indices.append(i)
                pos_indices.append(pos)
                wt_token_ids.append(self.seq_tokenizer.convert_tokens_to_ids(wt_char))
                mt_token_ids.append(self.seq_tokenizer.convert_tokens_to_ids(mut_char))

        b_idx = torch.tensor(batch_indices, dtype=torch.long)
        p_idx = torch.tensor(pos_indices, dtype=torch.long)
        wt_ids = torch.tensor(wt_token_ids, dtype=torch.long)
        mt_ids = torch.tensor(mt_token_ids, dtype=torch.long)

        return {
            "id": ids,
            "nsentences": len(samples),
            "net_input": {
                "wt": {
                    "seqs": wt_seq_inputs,
                    "stus": wt_stu_inputs,
                    "mlm_seqs": mlm_seq_inputs,
                },
                "inputs": {
                    "batch_idx": b_idx,
                    "pos_idx": p_idx,
                    "wt_ids": wt_ids,
                    "mt_ids": mt_ids,
                    "seq_mask_id": self._seq_m_id,
                    "bs": bs
                }
            },
            "target": {
                "labels": torch.tensor(labels, dtype=torch.float32),
                "mlm_masks": mlm_mask
            }
        }

    def num_tokens(self, index):
        return int(self.sizes[index])

    def size(self, index):
        return int(self.sizes[index])

    def ordered_indices(self):
        return np.random.permutation(len(self)).astype(np.int64)

    @property
    def supports_prefetch(self):
        return False


@register_task("fla_inp_tasks")
class stu_seq_tasks(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--data", dest="data_opt", default=None, help="")
        parser.add_argument("--split_name", default="standard_0", help="")
        parser.add_argument("--few_shot_size", type=int, default=None, help="Few-shot sample size (8, 16, 32, 64, 128, 256)")
        parser.add_argument("--few_shot_seed", type=int, default=42, help="Random seed for few-shot sampling")

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.few_shot_size = getattr(args, "few_shot_size", None)
        self.few_shot_seed = getattr(args, "few_shot_seed", 42)

        self.data = pd.read_csv(dms_csv_path)

        np.random.seed(self.few_shot_seed)
        indices = np.random.choice(len(self.data), size=self.few_shot_size, replace=False)
        self.train_subset = self.data.iloc[indices]
        self.test_subset = self.data.drop(indices)

        # Buffers for collecting validation predictions across batches
        self._valid_preds = []
        self._valid_targets = []

        print(f"Loaded DMS dataset from {dms_csv_path}, train sample: {self.train_subset.shape}; test sample: {self.test_subset.shape}")

    def _load_split_file(self, split_path):

        if split_path == "train":
            data_subset = self.train_subset
        else:
            data_subset = self.test_subset

        stus, muts, scores = data_subset["ss"].values.tolist(), data_subset["mut"].values.tolist(), data_subset["label"].values.tolist()
        return SSDataset(stus, muts, scores)


    def load_dataset(self, split, **kwargs):
        self.datasets[split] = self._load_split_file(split)

    def build_model(self, args):
        from fairseq.models import build_model as _build_model
        return _build_model(args, self)

    @property
    def target_dictionary(self):
        class _DummyDict:
            def cls(self): return 0
            def pad(self): return 1
            def eos(self): return 2
            def unk(self): return 3
            def mask(self): return 4
        return _DummyDict()

    def begin_valid_epoch(self, epoch, model):
        """Clear prediction buffers before each validation epoch."""
        self._valid_preds = []
        self._valid_targets = []

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if "predictions" in logging_output:
            self._valid_preds.append(logging_output["predictions"])
            self._valid_targets.append(logging_output["targets"])
        return loss, sample_size, logging_output

    def post_validate(self, model, stats, agg):
        """Compute epoch-level PCC/SCC after all validation batches."""
        if len(self._valid_preds) > 0:
            all_preds = torch.cat(self._valid_preds, dim=0).numpy()
            all_targets = torch.cat(self._valid_targets, dim=0).numpy()
            result = eval_reg(all_targets, all_preds)
            stats["RMSE"] = round(result["RMSE"], 4)
            stats["PCC"] = round(result["PCC"], 4)
            stats["SCC"] = round(result["SCC"], 4)
            self._valid_preds = []
            self._valid_targets = []
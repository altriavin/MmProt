import os
from typing import List
import numpy as np
import torch
from fairseq.tasks import FairseqTask, register_task
from fairseq.data import FairseqDataset, data_utils
from transformers import EsmTokenizer, EsmForMaskedLM

import pandas as pd
from pkg_seq_stu.constants import esm_root_path, saport_root_path, data_root_path, AA_TO_IDX
from .data_augment import mlm_seq_pred

class SSDataset(FairseqDataset):
    def __init__(self, stus, seqs):
        self.stus = stus
        self.seqs = seqs

        self.sizes = np.fromiter((len(x) for x in self.seqs), dtype=np.int64)

        self.seq_tokenizer = EsmTokenizer.from_pretrained(
            esm_root_path, weights_only=False
        )
        self.stu_tokenizer = EsmTokenizer.from_pretrained(
            saport_root_path, weights_only=False
        )

    def __getitem__(self, index):
        return {"id": index, "seq": self.seqs[index], "stu": self.stus[index]}

    def __len__(self):
        return len(self.seqs)

    def collater(self, samples):

        if len(samples) == 0:
            return {}

        ids, seqs, stus = [], [], []
        for item in samples:
            ids.append(item["id"])

            seqs.append(item["seq"])
            stus.append(item["stu"])

        mlm_src, mlm_tgt, mlm_mask = mlm_seq_pred(seqs)

        src_inputs = self.seq_tokenizer(
            mlm_src,
            padding=True,
            return_tensors="pt"
        )
        tgt_inputs = self.seq_tokenizer(
            mlm_tgt,
            padding=True,
            return_tensors="pt"
        )
        stu_inputs = self.stu_tokenizer(
            stus,
            padding=True,
            return_tensors="pt"
        )

        return {
            "id": ids,
            "nsentences": len(samples),
            "net_input": {
                "inputs": {
                    "src_inputs": src_inputs,
                    "stu_inputs": stu_inputs,
                }
            },
            "target": {
                "tgt_inputs": tgt_inputs["input_ids"],
                "mlm_masks": mlm_mask.bool()
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

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.data_path = getattr(args, "data_opt", None)

    def _load_split_file(self, split_path):
        data_subset = pd.read_csv(split_path)
        stus, seqs = data_subset["ss"].values.tolist(), data_subset["seq"].values.tolist()
        return SSDataset(stus, seqs)

    def load_dataset(self, split, **kwargs):
        if split != "train":
            return
        path = os.path.join(data_root_path, f"{self.data_path}.csv")
        self.datasets[split] = self._load_split_file(path)

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
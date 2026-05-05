import os
from typing import List
import numpy as np
import torch
from fairseq.tasks import FairseqTask, register_task
from fairseq.data import FairseqDataset, data_utils

import esm
from Bio import SeqIO
from transformers import EsmTokenizer, EsmForMaskedLM
import lmdb
import json

from pkg_seq_stu.constants import esm_root_path, saport_root_path
from .data_augment import seq_aug, stu_aug

class SSDataset(FairseqDataset):
    def __init__(self, seq_seqs, stu_seqs):
        self.seq_seqs = seq_seqs
        self.stu_seqs = stu_seqs

        self.sizes = np.fromiter((len(x) for x in self.seq_seqs), dtype=np.int64)

        self.seq_tokenizer = EsmTokenizer.from_pretrained(
            esm_root_path, weights_only=False
        )

        self.stu_tokenizer = EsmTokenizer.from_pretrained(
            saport_root_path, weights_only=False
        )

    def __getitem__(self, index):
        return {"id": index, "seq_seq": self.seq_seqs[index], "stu_seq": self.stu_seqs[index]}

    def __len__(self):
        return len(self.seq_seqs)
    
    def collater(self, samples):
        
        if len(samples) == 0:
            return {}

        ids, seq_seq, stu_seq = [], [], []
        for item in samples:
            ids.append(item["id"])

            seq = item["seq_seq"]
            stu = item["stu_seq"]

            seq_seq.append(seq)
            stu_seq.append(stu)

        aug_seq_seq = [seq_aug(seq) for seq in seq_seq]
        aug_stu_seq = [stu_aug(seq) for seq in stu_seq]

        seq_inputs = self.seq_tokenizer(
            seq_seq,
            padding=True,
            return_tensors="pt"
        )
        aug_seq_inputs = self.seq_tokenizer(
            aug_seq_seq,
            padding=True,
            return_tensors="pt"
        )
        
        stu_inputs = self.stu_tokenizer(
            stu_seq,
            padding=True,
            return_tensors="pt"
        )
        aug_stu_inputs = self.stu_tokenizer(
            aug_stu_seq,
            padding=True,
            return_tensors="pt"
        )

        seq_masks = seq_inputs['attention_mask'].eq(0)
        stu_masks = stu_inputs['attention_mask'].eq(0)

        return {
            "id": ids,
            "nsentences": len(samples),
            "net_input": {
                "seq_seqs": seq_inputs,
                "aug_seq_seqs": aug_seq_inputs,
                "stu_seqs": stu_inputs,
                "aug_stu_seqs": aug_stu_inputs,
                "seq_masks": seq_masks,
                "stu_masks": stu_masks
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


def load_data_from_fasta(split_path):
    seq_seqs, stu_seqs = [], []

    with lmdb.open(split_path, readonly=True, lock=False).begin() as txn:
        length = int(txn.get('length'.encode()).decode())
        for i in range(length):
            data_str = txn.get(str(i).encode()).decode()
            data = json.loads(data_str)
            seq_seq = data["seq"][::2]
            seq_stu = data["seq"]

            seq_seqs.append(seq_seq)
            stu_seqs.append(seq_stu)
    return seq_seqs, stu_seqs


@register_task("fla_align_tasks")
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
        seq_seqs, stu_seqs = load_data_from_fasta(split_path)
        return SSDataset(seq_seqs, stu_seqs)

    def load_dataset(self, split, **kwargs):
        path = os.path.join(self.data_path, f"{split}")
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
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
    def __init__(self, stus, seqs, muts, scores):
        self.stus = stus
        self.seqs = seqs
        self.muts = muts
        self.labels = scores

        self.sizes = np.fromiter((len(x) for x in self.seqs), dtype=np.int64)

        self.seq_tokenizer = EsmTokenizer.from_pretrained(
            esm_root_path, weights_only=False
        )
        self.stu_tokenizer = EsmTokenizer.from_pretrained(
            saport_root_path, weights_only=False
        )

    def __getitem__(self, index):
        return {"id": index, "seq": self.seqs[index], "stu": self.stus[index], "mut": self.muts[index], "label": self.labels[index]}

    def __len__(self):
        return len(self.seqs)

    def collater(self, samples):

        if len(samples) == 0:
            return {}

        ids, wt_seq_seqs, wt_seq_stus = [], [], []
        muts, labels = [], []
        for item in samples:
            ids.append(item["id"])

            wt_seq_seqs.append(item["seq"])
            wt_seq_stus.append(item["stu"])
            muts.append(item["mut"])
            labels.append(item["label"])

        mlm_seq, _, mlm_mask = mlm_seq_pred(wt_seq_seqs)

        wt_seq_inputs = self.seq_tokenizer(
            wt_seq_seqs,
            padding=True,
            return_tensors="pt"
        )
        mlm_seq_inputs = self.seq_tokenizer(
            mlm_seq,
            padding=True,
            return_tensors="pt"
        )
        mlm_seq_inputs['attention_mask'] = wt_seq_inputs["attention_mask"]

        wt_stu_inputs = self.stu_tokenizer(
            wt_seq_stus,
            padding=True,
            return_tensors="pt"
        )

        seq_m_id = self.seq_tokenizer.mask_token_id
        batch_indices = []
        pos_indices = []
        wt_token_ids = []
        mt_token_ids = []

        for i, mut_str in enumerate(muts):
            for m in mut_str.split(','):
                wt_char = m[0]
                pos = int(m[1:-1]) - 1
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
                    "seq_mask_id": seq_m_id,
                    "bs": len(muts)
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

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.data_path = getattr(args, "data_opt", None)

    def _load_split_file(self, split_path):
        data_subset = pd.read_csv(split_path)
        stus, seqs, muts, scores = data_subset["ss"].values.tolist(), data_subset["seq"].values.tolist(), data_subset["mut_type"].values.tolist(), data_subset["ddG_ML"].values.tolist()
        return SSDataset(stus, seqs, muts, scores)

    def load_dataset(self, split, **kwargs):
        path = os.path.join(data_root_path, f"{self.data_path}_{split}.csv")
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
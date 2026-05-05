import random
import numpy as np

import torch

from pkg_seq_stu.constants import AA_ORDER
# AA_ORDER = "LAGVSERTIDPKQNFYMHWC"

from typing import Dict, Iterable, List, Tuple, Union


def seq_aug(
    seq: str,
    p: float = 0.22,
    *,
    mapping: Dict[str, str] = None,
    do_not_sub: Iterable[str] = ("P", "G", "W", "H"),
    seed: Union[int, None] = None,
    strict: bool = False,
    stochastic_round: bool = True,
) -> Tuple[str, List[int]]:
    AA20 = set("ACDEFGHIKLMNPQRSTVWY")
    if strict:
        bad = [c for c in seq if c not in AA20]
        if bad:
            raise ValueError(f"序列含非 20aa 字符: {set(bad)}")

    if mapping is None:
        mapping = {
            "A": "V",
            "S": "T",
            "F": "Y",
            "K": "R",
            "C": "M",
            "D": "E",
            "N": "Q",
            "V": "I",
        }

    rng = random.Random(seed)
    seq_list = list(seq)

    dns = set(do_not_sub)
    eligible = [i for i, aa in enumerate(seq_list)
                if (aa not in dns) and (aa in mapping) and (mapping[aa] != aa)]
    n = len(eligible)

    if n == 0 or p <= 0.0:
        return seq
    if p >= 1.0:
        chosen = eligible
    else:
        target = p * n
        if stochastic_round:
            base = int(target)
            prob_plus_one = target - base
            k = base + (1 if rng.random() < prob_plus_one else 0)
        else:
            k = int(round(target))
        k = max(0, min(n, k))
        chosen = rng.sample(eligible, k)

    changed = []
    for i in chosen:
        aa = seq_list[i]
        new_aa = mapping.get(aa, aa)
        if new_aa != aa:
            seq_list[i] = new_aa
            changed.append(i)

    return "".join(seq_list)


def stu_aug(
    seq: str,
    p: float = 0.15,
    *,
    token_len: int = 2,
    mask_char: str = "#",
    seed: Union[int, None] = None,
    stochastic_round: bool = True,
    exclude_already_masked: bool = True, 
) -> Tuple[str, List[int]]:
    tokens = [seq[i:i+token_len] for i in range(0, len(seq), token_len)]

    if exclude_already_masked:
        eligible = [i for i, t in enumerate(tokens) if t[0] != mask_char]
    else:
        eligible = list(range(len(tokens)))
    n = len(eligible)

    if n == 0 or p <= 0.0:
        return seq, []

    rng = random.Random(seed)
    if p >= 1.0:
        chosen = eligible
    else:
        target = p * n
        base = int(target)
        k = base + (1 if stochastic_round and rng.random() < (target - base) else 0)
        k = max(0, min(n, k))
        chosen = rng.sample(eligible, k)

    for i in chosen:
        t = tokens[i]
        tokens[i] = mask_char + t[1:]

    return "".join(tokens)


def mlm_seq_pred(sequences):
    tgt = list(sequences[:])
    src = []
    mask = []
    for seq in sequences:
        if len(seq) == 0:
            tgt.remove(seq)
            continue
        mod_idx = random.sample(list(range(len(seq))), int(len(seq) * 0.15))
        if len(mod_idx) == 0:
            mod_idx = [np.random.choice(len(seq))]
        seq_mod = list(seq)
        for idx in mod_idx:
            p = np.random.uniform()
            if p <= 0.10:
                mod = seq[idx]
            elif 0.10 < p <= 0.20:
                mod = np.random.choice([i for i in AA_ORDER if i != seq[idx]])
            else:
                mod = "<mask>"
            seq_mod[idx] = mod
        src.append(''.join(seq_mod))
        m = torch.zeros(len(seq_mod))
        m[mod_idx] = 1
        m = torch.cat([torch.zeros(1), m, torch.zeros(1)])
        mask.append(m)
    return src, tgt, torch.stack(mask)


if __name__ == "__main__":
    # seq = "ACDEFGHIKLMNPQRSTVWY"
    # new_seq = replace_dictionary_ratio(seq, p=0.5)
    # print(f"seq: {seq}")
    # print(f"aug seq: {new_seq}")

    # seq = "M#EvVpQpL#VyQdYaKv"
    # new_seq = stu_aug(seq, p=0.3, token_len=2, mask_char="#")
    # print(f"seq: {seq}")
    # print(f"aug seq: {new_seq}")

    seqs = ["ACDEFGHIKLMNPQRSTVWY"]
    src, tgt, masks = mlm_seq_pred(seqs, 8)
    print(f'src: {src}')
    print(f'tgt: {tgt}')
    print(f'masks: {masks}')
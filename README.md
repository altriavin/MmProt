# MmProt

MmProt is a multi-modal protein representation learning framework that fuses protein sequence and structure information for mutation effect prediction. It adopts a multi-stage training strategy: cross-modal alignment, masked language model fine-tuning, and downstream task adaptation.

## Architecture

MmProt uses ESM-2 (650M) as the sequence encoder and SaProt (650M) as the structure encoder. A Flash Attention-based BERT Fusion Encoder performs cross-modal fusion.

```
Sequence  ──► ESM-2 (frozen) ──► Seq Projection ──┐
                                                   ├──► Fusion Encoder (BERT + Flash Attention) ──► Output
Structure ──► SaProt (frozen) ──► Stu Projection ──┘
```

## Training Stages

### Stage I: Cross-Modal Alignment Pre-training

Aligns sequence and structure representations via contrastive learning. Data augmentation (synonymous amino acid substitution, structure token masking, embedding rotation noise) generates multi-view representations. InfoNCE loss pulls together different views of the same protein.

- Data format: LMDB containing SaProt-style structure-sequence tokens
- Training: Multi-node distributed (torchrun, 2 nodes x 8 GPUs)
- Loss: InfoNCE contrastive loss

```bash
cd stage_I
bash pretrain.sh
```

### Stage II: Masked Language Model Fine-tuning

Performs MLM training on the aligned Fusion Encoder from Stage I to enhance protein sequence understanding.

- Data format: CSV with `seq` (amino acid sequence) and `ss` (SaProt structure sequence)
- Training: Single-node multi-GPU (torchrun)
- Loss: Masked Cross-Entropy

```bash
cd stage_II
bash train.sh <gpu_ids> <data_name>
```

### Stage III: Few-shot DMS Mutation Effect Prediction

Loads Stage II pre-trained weights and fine-tunes on Deep Mutational Scanning (DMS) data in a few-shot setting. The model scores mutation sites via a mask-then-predict strategy, trained with Bradley-Terry ranking loss and MLM auxiliary loss.

- Data format: CSV with `ss` (structure sequence), `mut` (mutation annotation, e.g. `A123V`), `label` (functional score)
- Training: Single GPU
- Loss: Bradley-Terry ranking loss + Masked Cross-Entropy
- Metrics: PCC, SCC, RMSE

```bash
cd stage_III
bash train_dms_fewshot.sh <gpu_ids> <few_shot_size> [beta] [gamma] [seed]
```

## Downstream Tasks

### Protein Fitness Prediction

Evaluates mutation fitness prediction on multiple DMS datasets, supporting data splits by mutation, position, regime, or score.

Supported datasets: avgfp, gb1, pab1, ube4b, tem-1, dlg4-2022-abundance, dlg4-2022-binding, grb2-abundance, grb2-binding, pten-abundance, pten-activity

```bash
cd protein_fitness_prediction
bash train_test_multi.sh <gpu_ids> <data_name> [split_prefix] [beta] [gamma]
```

### Protein Stability Prediction

Predicts the effect of mutations on protein thermodynamic stability (ddG). Runs 3 times with different random seeds and reports averaged results.

```bash
cd protein_stability_prediction
bash train_test_seed.sh <gpu_ids> <data_name> [test_name] [beta] [gamma]
```

## Installation

### Dependencies

- Python 3.10+
- PyTorch 2.4.1+cu121
- fairseq
- transformers
- flash-attn
- einops
- biopython
- lmdb
- pandas

### Setup

```bash
# Downgrade pip first (required for fairseq installation)
pip install pip==24.0

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install fairseq transformers flash-attn einops biopython lmdb pandas
```

> **Note:** pip must be downgraded to 24.0 before installing fairseq, otherwise the installation may fail.

### Pre-trained Models

The framework depends on the following pre-trained models, which are automatically downloaded from HuggingFace on first run:

- **ESM-2**: `facebook/esm2_t33_650M_UR50D`
- **SaProt**: `westlake-repl/SaProt_650M_AF2`

For offline environments, download the models in advance and update the paths in `constants.py`.

## Project Structure

```
MmProt/
├── stage_I/                          # Stage I: Cross-modal alignment
│   ├── pretrain.sh
│   └── pkg_seq_stu/
│       ├── constants.py
│       ├── models/
│       │   ├── seq_stu_model.py      # Alignment model (InfoNCE)
│       │   ├── seq_encoder.py        # ESM-2 encoder
│       │   ├── stu_encoder.py        # SaProt encoder
│       │   └── fla_attn/             # Flash Attention BERT
│       ├── criterions/
│       │   └── seq_stu_loss.py       # InfoNCE contrastive loss
│       └── tasks/
│           ├── seq_stu_tasks.py      # LMDB data loading
│           └── data_augment.py       # Sequence/structure augmentation
│
├── stage_II/                         # Stage II: MLM fine-tuning
│   ├── train.sh
│   └── pkg_seq_stu/
│       ├── models/
│       │   └── task_sft_model.py     # MLM prediction head
│       ├── criterions/
│       │   └── seq_stu_loss.py       # Masked Cross-Entropy
│       └── tasks/
│           ├── seq_stu_tasks.py      # CSV data loading
│           └── data_augment.py       # MLM masking
│
├── stage_III/                        # Stage III: Few-shot DMS
│   ├── train_dms_fewshot.sh
│   └── pkg_seq_stu/
│       ├── models/
│       │   ├── few_model.py          # Few-shot model (cached inference)
│       │   └── task_sft_model.py     # Mutation scoring model
│       ├── criterions/
│       │   └── seq_stu_loss.py       # BT ranking + MLM loss
│       └── tasks/
│           └── seq_stu_tasks.py      # DMS few-shot data loading
│
├── protein_fitness_prediction/       # Downstream: fitness prediction
│   ├── train_test_multi.sh
│   └── pkg_seq_stu/
│
└── protein_stability_prediction/     # Downstream: stability prediction (ddG)
    ├── train_test_seed.sh
    └── pkg_seq_stu/
```

# MmProt: Structure-Prompted Multimodal Protein Language Model

Official repository for **MmProt**, a **M**ulti**m**odal **Prot**ein Language Model for structure-prompted, preference-aligned fitness prediction under limited experimentally grounded supervision.

MmProt integrates sequence and structure information through multimodal contrastive pre-training, target-specific evolutionary specialization, and preference-based fitness adaptation. By using the wild-type structure as a structural prompt, MmProt aims to better model mutation effects when only limited DMS measurements are available.

## Overview

MmProt consists of three main stages:

1. **Stage I: Multimodal Contrastive Pre-training**  
   MmProt is pretrained on large-scale protein sequence--structure pairs to learn informative sequence--structure representations in a shared latent space.

2. **Stage II: Structure-Prompted Evolutionary Specialization**  
   For each target protein, MmProt uses the wild-type structure as a fixed structural prompt and performs masked language modeling on naturally occurring related variants.

3. **Stage III: Structure-Prompted Fitness Prediction**  
   When limited DMS measurements are available, MmProt performs preference-based fine-tuning to align model-induced variant rankings with experimental fitness preferences.

For protein structures, Foldseek is used to convert structural information into token sequences before training and inference.

## Installation

### Setup

```bash
# Downgrade pip first. This is required for fairseq installation.
pip install pip==24.0

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install fairseq
pip install pandas
pip install transformers
pip install einops
pip install biopython
pip install lmdb
```

> **Note:** `pip` should be downgraded to version `24.0` before installing `fairseq`, otherwise the installation may fail.

## Pretrained Model Dependencies

MmProt depends on the following pretrained protein language models, which are automatically downloaded from HuggingFace on the first run:

- ESM-2: `facebook/esm2_t33_650M_UR50D`
- SaProt: `westlake-repl/SaProt_650M_AF2`

For offline environments, please download these models in advance and update the corresponding paths in `constants.py`.

## Quick Start

### Stage I: Multimodal Contrastive Pre-training

Stage I performs large-scale sequence--structure contrastive pre-training.

```bash
cd stage_I
bash pretrain.sh
```

The pre-training data follows the SaProt-style sequence--structure processing pipeline. In our experiments, Stage I was trained on 40 million protein sequence--structure pairs using 16 A800 GPUs for approximately one month.

> Due to conference anonymity requirements, pretrained MmProt checkpoints will be released after paper acceptance.

### Stage II: Structure-Prompted Evolutionary Specialization

Stage II performs target-specific masked language modeling using naturally occurring related variants and the wild-type structure as a structural prompt.

```bash
cd stage_II
bash train.sh
```

The trained checkpoint will be saved to:

```text
/root/autodl-tmp/MmProt/stage_II/toy/checkpoint_last.pt
```

### Stage III: Few-shot DMS Adaptation

Stage III performs preference-based adaptation using limited DMS measurements.

```bash
cd stage_III
bash train_dms_fewshot.sh 0 8
```

Arguments:

```text
0  -> GPU ID
8  -> number of labeled variants
```

## Downstream Experiments

### Protein Fitness Prediction

```bash
cd protein_fitness_prediction
bash train_test_multi.sh
```

### Protein Stability Prediction

```bash
cd protein_stability_prediction
bash train_test_seed.sh
```

## Model Checkpoints and Processed Data

The MmProt pretrained weights, stage-specific checkpoints, and processed datasets used in this project will be made publicly available after the paper is accepted.

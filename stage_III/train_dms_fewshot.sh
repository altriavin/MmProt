#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_IDS="${1:-0}"
FEW_SHOT_SIZE="${2:?Usage: $0 <gpu_ids> <few_shot_size> [beta] [gamma] [seed]}"
BETA="${3:-0.9}"
GAMMA="${4:-0.1}"
SEED="${5:-42}"

export NUMEXPR_MAX_THREADS=32
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTHONWARNINGS=ignore

echo "[INFO] Using GPU(s): ${GPU_IDS}"
echo "[INFO] Few-shot size: ${FEW_SHOT_SIZE}"
echo "[INFO] Using Beta: ${BETA} and Gamma: ${GAMMA}"
echo "[INFO] Using Seed: ${SEED}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" fairseq-train \
  --few_shot_size "${FEW_SHOT_SIZE}" \
  --few_shot_seed "${SEED}" \
  --train-subset train \
  --valid-subset test \
  --user-dir pkg_seq_stu \
  --task fla_inp_tasks \
  --arch fla_inp_model_base \
  --criterion fla_inp_criterion \
  --batch-size 128 \
  --max-tokens 12800 \
  --temperature 2.0 \
  --beta "${BETA}" \
  --gamma "${GAMMA}" \
  --label-temperature 0.1 \
  --task_loss_weight 0.4 \
  --num_hidden_layers 12 \
  --num_attention_heads 8 \
  --hidden_size 1024 \
  --head_dim 128 \
  --num-workers 8 \
  --optimizer adam \
  --adam-betas '(0.9,0.98)' \
  --lr 4e-5 \
  --lr-scheduler inverse_sqrt \
  --clip-norm 1.0 \
  --warmup-updates 4000 \
  --warmup-init-lr 1e-7 \
  --ddp-backend legacy_ddp \
  --save-dir tmp \
  --skip-invalid-size-inputs-valid-test \
  --keep-interval-updates 1 \
  --save-interval 1 \
  --keep-best-checkpoints 1 \
  --best-checkpoint-metric loss \
  --log-format tqdm \
  --log-interval 10000 \
  --fp16 \
  --no-epoch-checkpoints \
  --required-batch-size-multiple 1 \
  --no-save \
  --max-epoch 200

echo "[DONE] Few-shot size: ${FEW_SHOT_SIZE}, seed: ${SEED}"


# --maximize-best-checkpoint-metric \
# --patience 15 \
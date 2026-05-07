#!/usr/bin/env bash
set -euo pipefail

# 位置参数
GPU_IDS="${1:-0}"            # 例如 0 或 0,1 或 0,1,2
DATA_NAME="${2:-toy}"        # 例如 gb1 / avgfp / dlg4-2022-binding ...

export NUMEXPR_MAX_THREADS=32
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTHONWARNINGS=ignore
export HF_HUB_OFFLINE=1

# 去掉可能的空格，然后按逗号切分，统计 GPU 数量
GPU_IDS_CLEAN="${GPU_IDS// /}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS_CLEAN}"
NPROC_PER_NODE="${#GPU_ARRAY[@]}"

echo "[INFO] Using GPU(s): ${GPU_IDS_CLEAN}"
echo "[INFO] Number of processes: ${NPROC_PER_NODE}"
echo "[INFO] Using dataset: ${DATA_NAME}"

CUDA_VISIBLE_DEVICES="${GPU_IDS_CLEAN}" torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  $(which fairseq-train) \
    --data "${DATA_NAME}" \
    --disable-validation \
    --user-dir pkg_seq_stu \
    --task fla_inp_tasks \
    --arch fla_inp_model_base \
    --criterion fla_inp_criterion \
    --batch-size 128 \
    --max-tokens 12800 \
    --num_hidden_layers 12 \
    --num_attention_heads 8 \
    --hidden_size 1024 \
    --num-workers 4 \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --lr 4e-5 \
    --log-interval 1 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 1.0 \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-7 \
    --ddp-backend legacy_ddp \
    --save-dir "toy" \
    --skip-invalid-size-inputs-valid-test \
    --keep-interval-updates 1 \
    --save-interval 1 \
    --keep-best-checkpoints 1 \
    --no-epoch-checkpoints \
    --best-checkpoint-metric loss \
    --log-format tqdm \
    --fp16 \
    --required-batch-size-multiple 1 \
    --max-epoch 50 \

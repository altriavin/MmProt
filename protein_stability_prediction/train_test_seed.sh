#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_IDS="${1:-0}"
DATA_NAME="${2:-gb1}"
TEST_NAME="${3:-test}"
BETA="${4:-0.5}"
GAMMA="${5:-0.5}"

shopt -s nullglob

export NUMEXPR_MAX_THREADS=32
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTHONWARNINGS=ignore

echo "[INFO] Using GPU(s): ${GPU_IDS}"
echo "[INFO] Using dataset: ${DATA_NAME}"
echo "[INFO] Using Beta: ${BETA} and Gamma: ${GAMMA}"

# --------- 生成 3 个不同的随机种子 ----------
mapfile -t SEEDS < <(python - <<'PY'
import os, random
random.seed(int.from_bytes(os.urandom(16), "little"))
seeds = random.sample(range(1, 2**31 - 1), 3)  # 3个互不相同的seed
print("\n".join(map(str, seeds)))
PY
)

echo "[INFO] Random seeds: ${SEEDS[*]}"
echo

# --------- 依次用不同 seed 跑 3 次 ----------
for idx in "${!SEEDS[@]}"; do
  SEED="${SEEDS[$idx]}"
  RUN_ID=$((idx + 1))

  echo "=============================================="
  echo "[RUN ${RUN_ID}/3] Dataset=${DATA_NAME} Beta=${BETA} Gamma=${GAMMA} Seed=${SEED}"
  echo "=============================================="

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" fairseq-train \
    --seed "${SEED}" \
    --data "${DATA_NAME}" \
    --train-subset train \
    --valid-subset "${TEST_NAME}" \
    --user-dir pkg_seq_stu \
    --task fla_inp_tasks \
    --arch fla_inp_model_base \
    --criterion fla_inp_criterion \
    --batch-size 120 \
    --max-tokens 12000 \
    --temperature 2.0 \
    --beta "${BETA}" \
    --gamma "${GAMMA}" \
    --label-temperature 0.1 \
    --task_loss_weight 0.4 \
    --num_hidden_layers 12 \
    --num_attention_heads 8 \
    --hidden_size 1024 \
    --head_dim 128 \
    --num-workers 4 \
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
    --best-checkpoint-metric PCC \
    --maximize-best-checkpoint-metric \
    --log-format tqdm \
    --log-interval 10000 \
    --fp16 \
    --no-epoch-checkpoints \
    --required-batch-size-multiple 1 \
    --max-epoch 50 \
    --no-save \

  echo "[DONE] run ${RUN_ID}/3 seed=${SEED}"
  echo
done

echo "[ALL DONE] Dataset=${DATA_NAME}"
echo "[ALL DONE] Seeds used: ${SEEDS[*]}"

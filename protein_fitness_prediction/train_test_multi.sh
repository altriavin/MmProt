# avgfp, dlg4-2022-abundance, gb1, grb2-binding, pten-abundance, ube4b,
# dlg4-2022-binding, grb2-abundance, pab1, pten-activity, tem-1

# mutation, position, regime, score


#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_IDS="${1:-0}"
DATA_NAME="${2:-gb1}"
SPLIT_PREFIX="${3:-mutation}"
BETA="${4:-0.5}"
GAMMA="${5:-0.5}"

DATA_ROOT="/root/huxiaowen/mut_llm/MmProt/protein_fitness_prediction/toy"

shopt -s nullglob

export NUMEXPR_MAX_THREADS=32
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTHONWARNINGS=ignore

echo "[INFO] Using GPU(s): ${GPU_IDS}"
echo "[INFO] Using dataset: ${DATA_NAME}"
echo "[INFO] Using split prefix: ${SPLIT_PREFIX}"
echo "[INFO] Using Beta: ${BETA} and Gamme: ${GAMMA}"

SPLIT_DIR_PATTERN="${DATA_ROOT}/${DATA_NAME}/${SPLIT_PREFIX}_*"
SPLIT_DIRS=(${SPLIT_DIR_PATTERN})

if [ "${#SPLIT_DIRS[@]}" -eq 0 ]; then
  echo "[ERROR] ${DATA_ROOT}/${DATA_NAME} not found the subfolder of ${SPLIT_PREFIX}_* "
  exit 1
fi

echo "[INFO] Found ${#SPLIT_DIRS[@]} split folders:"
for d in "${SPLIT_DIRS[@]}"; do
  echo "  - $(basename "$d")"
done
echo

for split_dir in "${SPLIT_DIRS[@]}"; do
  SPLIT_NAME="$(basename "${split_dir}")"

  # if [[ "${SPLIT_NAME}" =~ ^mutation_([0-1])$ ]]; then
  #   echo "[SKIP] ${DATA_NAME} - ${SPLIT_NAME}"
  #   echo
  #   continue
  # fi

  echo "=============================================="
  echo "[RUN] Dataset: ${DATA_NAME}, Split: ${SPLIT_NAME}"
  echo "=============================================="

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" fairseq-train \
    --data "${DATA_NAME}" \
    --train-subset train \
    --valid-subset test \
    --split_name "${SPLIT_NAME}" \
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
    --best-checkpoint-metric SCC \
    --maximize-best-checkpoint-metric \
    --log-format tqdm \
    --log-interval 10000 \
    --fp16 \
    --no-epoch-checkpoints \
    --required-batch-size-multiple 1 \
    --no-save \
    --max-epoch 100 \

  echo "[DONE] ${DATA_NAME} - ${SPLIT_NAME}"
  echo
done

echo "[ALL DONE] Dataset=${DATA_NAME}, Split prefix=${SPLIT_PREFIX}"

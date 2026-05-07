export NUMEXPR_MAX_THREADS=32
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

export PYTHONWARNINGS=ignore
export NODE_RANK="${NODE_RANK:-${RANK:-${REPLICA_INDEX:-0}}}"
export MASTER_PORT="${MASTER_PORT:-29400}"

torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  $(which fairseq-train) \
    --data /root/autodl-tmp/MmProt/stage_I/toy \
    --user-dir pkg_seq_stu \
    --task fla_align_tasks \
    --arch fla_align_model_base \
    --criterion fla_align_criterion \
    --batch-size 128 \
    --max-tokens 25600 \
    --num_hidden_layers 12 \
    --num_attention_heads 8 \
    --hidden_size 1024 \
    --num-workers 4 \
    --temperature 0.07 \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --lr 3e-4 \
    --log-interval 100 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 1.0 \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-7 \
    --ddp-backend legacy_ddp \
    --skip-invalid-size-inputs-valid-test \
    --save-dir checkpoint_fla_8_12 \
    --save-interval-updates 2000 \
    --keep-interval-updates 1 \
    --save-interval 1 \
    --keep-best-checkpoints 1 \
    --best-checkpoint-metric loss \
    --log-format tqdm \
    --fp16 \
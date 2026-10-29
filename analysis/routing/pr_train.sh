#!/bin/bash
# B3 pilot arm launcher (A83), direct-launch variant (no ray job submit).
# Usage: pr_train.sh <vanilla|cert09> <seed>
# Config surface byte-equivalent to repro/train_1gpu.sh except entry module
# e1.pr.main_pr and PR env vars. Checkpoints on internal NVMe.
set -euo pipefail

ARM="${1:?Usage: $0 <vanilla|cert09> <seed>}"
SEED="${2:?Usage: $0 <arm> <seed>}"

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../unlearnability-rlvr" && pwd)"
cd "$REPO_ROOT"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
PY=python

MODELS_ROOT="$(cd "$REPO_ROOT/.." && pwd)/models"
CKPT_ROOT="/home/rmedu/pr_b3_ckpts"
RUN_NAME="pr_${ARM}_seed${SEED}"
OUTDIR="$CKPT_ROOT/$RUN_NAME"
mkdir -p "$OUTDIR" logs

export WANDB_MODE=offline
export WANDB_DIR="$REPO_ROOT/wandb"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_ATTENTION_BACKEND=XFORMERS
export E1_PR_ARM="$ARM"
export E1_PR_EPS=0.75
export E1_PR_C=10
export E1_PR_R=15
export E1_PR_DELTA=0.90
export E1_PR_LOG="$OUTDIR/pr_telemetry.jsonl"

TARGET_STEP=120
MAX_PROMPT_LENGTH=1024
max_num_batched_tokens=$((MAX_PROMPT_LENGTH + 1024 + 1000))

cleanup() {
    echo "[watchdog] cleaning up"
    ray stop --force >/dev/null 2>&1 || true
    pkill -9 -f 'ray::' 2>/dev/null || true
    pkill -9 -f gcs_server 2>/dev/null || true
    if [ -n "${GCS_PRESERVE:-}" ] && [ -n "$(ls -A /tmp/ray 2>/dev/null)" ]; then
        mv /tmp/ray "$OUTDIR/failed_ray_logs_$(date +%H%M%S)" 2>/dev/null || true
    else
        rm -rf /dev/shm/ray* /tmp/ray* 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

ray stop --force >/dev/null 2>&1 || true
pkill -9 -f 'ray::' 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
for _ in $(seq 45); do
    ss -ltn 2>/dev/null | grep -q ':6379 ' || break
    sleep 1
done
rm -rf /tmp/ray/* /dev/shm/ray* 2>/dev/null || true

$PY -m e1.pr.main_pr \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.en_ctrl.type=fixed \
  actor_rollout_ref.actor.en_ctrl.en_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.en_ctrl.min_en_coef=0.001 \
  actor_rollout_ref.actor.en_ctrl.beta=0.01 \
  data.seed="$SEED" \
  data.train_files="$REPO_ROOT/data/SimpleRL-Zoo-Data/simplelr_qwen_level1to4_sub1k/train.parquet" \
  data.val_files="$REPO_ROOT/data/SimpleRL-Zoo-Data/simplelr_qwen_level1to4_sub1k/test.parquet" \
  data.train_batch_size=32 \
  data.val_batch_size=200 \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=1024 \
  data.replay=False \
  data.replay_n=1 \
  data.keep_old_advantage_before_downsample=False \
  data.keep_all_negative=False \
  actor_rollout_ref.model.path="$MODELS_ROOT/Qwen2.5-0.5B" \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.0001 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.low_clip_ratio=0.2 \
  actor_rollout_ref.actor.high_clip_ratio=0.2 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.oversample_n=-1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
  actor_rollout_ref.rollout.micro_rollout_batch_size=128 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.0001 \
  critic.ppo_micro_batch_size_per_gpu=4 \
  trainer.critic_warmup=0 \
  "trainer.logger=['console','wandb']" \
  trainer.project_name=b3_pr_pilot \
  trainer.remove_previous_ckpt=False \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.remove_clip=False \
  trainer.save_freq=$TARGET_STEP \
  trainer.test_freq=100000 \
  trainer.default_local_dir="$CKPT_ROOT/$RUN_NAME" \
  trainer.resume_data_state=True \
  trainer.total_epochs=12 &
TRAIN_PID=$!

echo "[watchdog] training pid=$TRAIN_PID target=global_step_${TARGET_STEP}"
STABLE=0
LAST_SIZE=-1
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    CKPT=$({ compgen -G "$OUTDIR/global_step_${TARGET_STEP}/actor/huggingface/model.safetensors" || true; } | head -1)
    if [ -n "$CKPT" ]; then
        SIZE=$(stat -c %s "$CKPT")
        if [ "$SIZE" -gt 0 ] && [ "$SIZE" = "$LAST_SIZE" ]; then
            STABLE=$((STABLE+1))
            if [ "$STABLE" -ge 2 ]; then
                echo "[watchdog] step-${TARGET_STEP} checkpoint complete -> stopping"
                kill -TERM "$TRAIN_PID" 2>/dev/null || true
                sleep 10
                kill -KILL "$TRAIN_PID" 2>/dev/null || true
                exit 0
            fi
        else
            STABLE=0
            LAST_SIZE=$SIZE
        fi
    fi
    sleep 30
done
wait "$TRAIN_PID" || true
echo "[watchdog] training exited without reaching step ${TARGET_STEP}"
exit 4

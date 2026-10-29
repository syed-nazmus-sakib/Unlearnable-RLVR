#!/bin/bash
#
# Single-GPU GRPO baseline driver: the repo's train_baseline.sh rescaled for
# 1x RTX 3090 Ti (24GB) instead of the paper's 4 GPUs. Pass a seed as $1.
#
# What changes vs train_baseline.sh (and why):
#   n_gpu_per_node   4    -> 1     only one GPU available
#   rollout_tp       2    -> 1     tensor parallel needs >=2 GPUs
#   train_batch_size 256  -> 32    fits 24GB alongside the colocated sglang engine
#   ppo_mini_batch   64   -> 16    keeps 2 optimizer steps per rollout batch, as in the paper
#   max_response_len 5120 -> 1024  KV cache + logprob activations on one card
#   ppo_micro_batch  2    -> 2     Qwen2.5's 151936-token vocab makes the LM-head logits
#   log_prob_micro   128  -> 4     the dominant allocation (~0.6MB fp32 per token), so the
#                                  micro batches are sized against logits, not params
#   dataset          full -> sub1k stratified 1023-prompt subset (repro/make_subset.py)
#   total_epochs     50   -> 12    see "step accounting" below
#
# Step accounting -- the two counters in ray_trainer.py are in DIFFERENT units, which
# is easy to get wrong:
#   * termination is `gen_steps >= total_training_steps` (ray_trainer.py:1079), and
#     total_training_steps = len(dataloader) * total_epochs. So it counts *generation*
#     batches (dataloader batches).
#   * save_freq / test_freq are `global_steps % freq` (ray_trainer.py:1314/1341), and
#     global_steps is the *optimizer* step counter -- the one in `global_step_120`.
# Dynamic sampling (the std>0 filter) discards prompts whose 8 rollouts all agree, then
# pulls another generation batch, so one optimizer step consumes ~3 generation batches.
# The paper's global_step_120 therefore corresponds to ~360 generation batches and
# ~11 exposures per prompt (120 * 3 * 256 / 8139), not ~4.
# Here: 12 epochs * 31 batches = 372 generation batches -> ~120 optimizer steps and
# ~11.6 exposures per prompt (372 * 32 / 1023). That matches the paper on both axes.
#
# rollout_n stays 8 and lr/kl/entropy stay at the paper's values -- those drive the
# GRPO signal the unlearnability analysis measures, so they are not scaled.

set -euo pipefail

seed="${1:?Usage: $0 <seed>}"

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO_ROOT"

# Absolute paths: `ray job submit` uploads WORKING_DIR and runs the entrypoint from a
# Ray temp dir, so train.sh's default relative MODELS_ROOT=../models would not resolve.
export MODELS_ROOT="${MODELS_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/models}"
export CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$REPO_ROOT/checkpoints}"
export HDFS_DATA_PATH="${HDFS_DATA_PATH:-$REPO_ROOT/data/SimpleRL-Zoo-Data}"
export PROJECT_NAME="${PROJECT_NAME:-unlearnability_rlvr}"
export WORKING_DIR="${WORKING_DIR:-$REPO_ROOT}"

# No WANDB_API_KEY here -> run offline. The trainer still needs wandb because
# ray_trainer.py reads metrics["actor/prompt_reward_stats"] as a wandb.Table
# to write prompt_reward_stats.csv.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/wandb}"

export RUN_NAME=qwen_0.5b_baseline_1gpu
export DATASET_NAME="${DATASET_NAME:-simplelr_qwen_level1to4_sub1k}"

#===============================================
# Ray setup with cleanup on exit
#===============================================
cleanup_on_exit() {
    echo "Cleaning up Ray cluster..."
    ray stop --force || true
    pkill -f ray || true
    rm -rf /dev/shm/ray* /tmp/ray* || true
}
trap cleanup_on_exit EXIT INT TERM

echo "Starting Ray cluster (1 GPU)..."
ray stop --force >/dev/null 2>&1 || true
ray start --head --num-gpus 1 --num-cpus "$(nproc)" --node-ip-address 0.0.0.0
export ARNOLD_WORKER_NUM=1

HYDRA_FULL_ERROR=1 bash train.sh \
    --model_name Qwen2.5-0.5B \
    --dataset_name "$DATASET_NAME" \
    --max_response_length 1024 \
    --train_batch_size 32 \
    --ppo_mini_batch_size 16 \
    --ppo_micro_batch_size 2 \
    --log_prob_micro_batch_size 4 \
    --micro_rollout_batch_size 128 \
    --rollout_n 8 \
    --rollout_tp 1 \
    --n_gpu_per_node 1 \
    --rollout_gpu_memory_util 0.45 \
    --kl_loss_coef 0.0001 \
    --entropy_coefficient 0.001 \
    --learning_rate 5e-7 \
    --save_freq 40 \
    --test_freq 100000 \
    --val_batch_size 200 \
    --total_epochs 12 \
    --rollout_name sglang \
    --resume_data_state True \
    --seed "$seed"
# save_freq 40 -> checkpoints at optimizer steps 40 / 80 / 120 (+ last), so
# global_step_120 exists as the paper's comparison point.
# test_freq is effectively disabled: repro/eval.sh does the pass@k evaluation that
# feeds classification, and the built-in validation loop would only cost GPU time.

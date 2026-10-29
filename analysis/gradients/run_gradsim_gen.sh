#!/bin/bash
# Figure 5 step 1: initial-policy rollouts for the gradient-similarity replication.
# N=200 at T=0.7 under the BASE model, matching Chen's grad-sim collection
# (results/qwen_0.5b_initial_grad_sim_0.7_200_...). His own 120-prompt set covers
# only 14 of our D_u 71 and 2 of the 10 rescued, so it cannot be reused.
set -uo pipefail
EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-1024}"
mkdir -p "$EXP/logs_gradsim"
cd "$REPO"
python -u inference/evaluate_model.py \
    --model-path "$(cd "$REPO/.." && pwd)/models/Qwen2.5-0.5B" \
    --tokenizer-path "$(cd "$REPO/.." && pwd)/models/Qwen2.5-0.5B" \
    --test-data "simplelr_qwen_level1to4_gradsim@train" --num-samples 200 \
    --max-new-tokens 1024 --temperature 0.7 \
    --prompt-type qwen --save-path "$REPO/results" \
    --model-name "e1_gradsim_initial" --seed 0 \
    --gpu-memory-util 0.60 --use-sglang \
    > "$EXP/logs_gradsim/gen.log" 2>&1
echo "gradsim generation done $(date '+%H:%M') rc=$?"

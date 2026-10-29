#!/bin/bash
#
# A49 — the registered 5120 budget re-run. Authorised at 2.7-3.75 h.
#
# Full 1023 prompts, N=128, cap 5120, TRAINED checkpoint (seed 1,
# global_step_120) -- identical to the A36 1024 reference in every respect
# except --max-new-tokens. That is the point: the only thing that changes is
# the evaluation budget.
#
# Sold as this and never as more (A49.5): it tests whether OUR central object
# is stable to the EVALUATION budget. It does NOT reproduce Chen's
# configuration, because our training also ran at 1024 and their policy was
# trained to use 5120.
#
# Primary: Jaccard of the guarded flagged set at 5120 against 1024, tau = 0.1,
# judged against the measured same-budget floor 0.8770 [0.8397, 0.9127] (A36).
#
# Chen's verifier call path is UNTOUCHED -- the parallel verifier is not
# substituted (A51.3). Verification is ~8 min at this checkpoint's measured
# 3.76 ms/response, so there is nothing to buy by substituting it.
set -uo pipefail

EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="$(cd "$EXP/../../unlearnability-rlvr" && pwd)"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-192}"

NAME="qwen_0.5b_ckpt1_n128_cap5120"
DATA="simplelr_qwen_level1to4_sub1k@train"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
LOG_DIR="$EXP/logs"; mkdir -p "$LOG_DIR"
cd "$REPO"

run_dir="$(ls -d "$REPO"/checkpoints/*_seed1 2>/dev/null | head -1)"
HF="$run_dir/global_step_120/actor/huggingface"
[[ -d "$HF" ]] || { echo "no checkpoint at $HF" >&2; exit 1; }

# A45 guard: stop while there is still room to stop. ~460 MB expected.
free_gb="$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9)"
(( free_gb >= 5 )) || { echo "only ${free_gb} GB free, refusing to start" >&2; exit 1; }

busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
[[ -z "$busy" ]] || { echo "GPU in use by PID(s): ${busy//$'\n'/ }" >&2; exit 1; }

# A45 open_new: refuse to overwrite an existing output.
if compgen -G "$REPO/results/${NAME}_${DATA}_*.json" >/dev/null; then
    echo "output already exists; refusing to overwrite" >&2; exit 1
fi

echo "######## A49 5120 re-run starting $(date '+%F %H:%M') ########"
echo "checkpoint: $HF"
echo "1023 prompts x 128 samples, cap 5120, temperature 0.7, prompt-type qwen"
python -u inference/evaluate_model.py \
    --model-path "$HF" --tokenizer-path "$HF" \
    --test-data "$DATA" --num-samples 128 \
    --max-new-tokens 5120 --temperature 0.7 \
    --prompt-type qwen --save-path "$REPO/results" \
    --model-name "$NAME" --seed 0 \
    --gpu-memory-util "$GPU_MEM_UTIL" --use-sglang
echo "rc=$?"
echo "######## A49 5120 re-run done $(date '+%F %H:%M') ########"

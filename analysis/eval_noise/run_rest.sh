#!/bin/bash
#
# Complete the 128-sample coverage: evaluate the 559 prompts the first
# high-sample pass did not touch, so pass@1 is known at 128 samples for all 1023.
#
#   bash run_rest.sh
#
# Same checkpoint (seed 1, global_step_120) and same sampling settings as the
# first pass, so the two result sets are directly poolable. ~77 min.
#
# Memory is unchanged from every other eval here: num_samples duplicates prompts
# into a flat list (gen_utils_sglang.py: all_prompts.extend([prompt] * n)) rather
# than becoming sglang's `n`, so the decode batch stays capped by
# SGLANG_MAX_RUNNING_REQUESTS regardless of 32 vs 128.

set -euo pipefail

EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-192}"

CKPT_SEED="${CKPT_SEED:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
MAX_LEN="${MAX_LEN:-1024}"
TEMP="${TEMP:-0.7}"
NAME="qwen_0.5b_ckpt${CKPT_SEED}_hi_rest"
DATA="simplelr_qwen_level1to4_rest@train"
LOG_DIR="$EXP/logs"; mkdir -p "$LOG_DIR"

cd "$REPO"

run_dir="$(ls -d "$REPO"/checkpoints/*_seed"${CKPT_SEED}" 2>/dev/null | head -1)"
HF="$run_dir/global_step_120/actor/huggingface"
[[ -d "$HF" ]] || { echo "no checkpoint at $HF" >&2; exit 1; }

# Refuse to fight another GPU job for VRAM. Ask the GPU directly rather than
# pgrep'ing for process names: any shell whose command line merely *mentions*
# the pattern (a launcher, a status check) matches `pgrep -f` and yields a false
# positive. --query-compute-apps lists only processes actually holding VRAM.
busy_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
if [[ -n "$busy_pids" ]]; then
    echo "GPU already in use by PID(s): ${busy_pids//$'\n'/ }" >&2
    echo "wait for that job or stop it first" >&2
    exit 1
fi

echo "building complement subset"
python "$EXP/make_rest_subset.py"

pat="$REPO/results/${NAME}_${DATA}_*.json"
if compgen -G "$pat" >/dev/null; then
    echo "already done: $(basename "$(ls $pat | head -1)")"
else
    echo "evaluating 559 prompts x 128 samples (~77 min)"
    set +e
    python -u inference/evaluate_model.py \
        --model-path "$HF" --tokenizer-path "$HF" \
        --test-data "$DATA" --num-samples 128 \
        --max-new-tokens "$MAX_LEN" --temperature "$TEMP" \
        --prompt-type qwen --save-path "$REPO/results" \
        --model-name "$NAME" --seed 0 \
        --gpu-memory-util "$GPU_MEM_UTIL" --use-sglang \
        > "$LOG_DIR/${NAME}.log" 2>&1
    rc=$?
    set -e
    grep -aE "Finished|pass@" "$LOG_DIR/${NAME}.log" | tail -3 || true
    if [[ $rc -ne 0 ]]; then
        echo "FAILED (exit $rc):" >&2; tail -15 "$LOG_DIR/${NAME}.log" >&2; exit "$rc"
    fi
    compgen -G "$pat" >/dev/null || { echo "exited 0 but wrote no results" >&2; exit 1; }
fi

echo
echo "done. next: python $EXP/final_du.py"

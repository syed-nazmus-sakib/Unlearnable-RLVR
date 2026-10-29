#!/bin/bash
#
# Final E0 run: 128-sample evaluation of SEED 2's checkpoint over all 1023 prompts.
#
# Purpose: seed 1's D_u (72 prompts) is defined on a single checkpoint, so the
# training-variation component of the instability is unmeasured at N=128. Running
# a second checkpoint at the same sample count isolates it: the cross-seed Jaccard
# at N=128 removes the evaluation noise that dominated the N=32 comparison
# (same-model 0.798 vs cross-seed 0.741), leaving only genuine training difference.
#
# Settings are identical to the seed-1 128-sample passes -- temperature 0.7,
# max_new_tokens 1024, prompt-type qwen, sglang -- so the two are directly
# comparable. The only difference is the checkpoint and the full 1023-prompt
# coverage in one pass.
#
# ~131k generations; at the 19-24 gen/s observed on this box, 1.5-2h.

set -euo pipefail

EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-192}"

CKPT_SEED=2
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
MAX_LEN=1024
TEMP=0.7
NAME="qwen_0.5b_ckpt${CKPT_SEED}_hi_all"
DATA="simplelr_qwen_level1to4_sub1k@train"
LOG_DIR="$EXP/logs"; mkdir -p "$LOG_DIR"

cd "$REPO"

run_dir="$(ls -d "$REPO"/checkpoints/*_seed"${CKPT_SEED}" 2>/dev/null | head -1)"
HF="$run_dir/global_step_120/actor/huggingface"
[[ -d "$HF" ]] || { echo "no checkpoint at $HF" >&2; exit 1; }

# Ask the GPU which PIDs hold VRAM rather than pgrep'ing for names: any shell
# whose command line merely mentions the pattern would be a false positive.
busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
if [[ -n "$busy" ]]; then
    echo "GPU already in use by PID(s): ${busy//$'\n'/ }" >&2; exit 1
fi

pat="$REPO/results/${NAME}_${DATA}_*.json"
if compgen -G "$pat" >/dev/null; then
    echo "already done: $(basename "$(ls $pat | head -1)")"; exit 0
fi

echo "seed ${CKPT_SEED} @ global_step_120, 1023 prompts x 128 samples"
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
grep -aE "Finished Generating|pass@1" "$LOG_DIR/${NAME}.log" | tail -2 || true
if [[ $rc -ne 0 ]]; then
    echo "FAILED (exit $rc):" >&2; tail -15 "$LOG_DIR/${NAME}.log" >&2; exit "$rc"
fi
compgen -G "$pat" >/dev/null || { echo "exited 0 but wrote no results" >&2; exit 1; }
echo "done."

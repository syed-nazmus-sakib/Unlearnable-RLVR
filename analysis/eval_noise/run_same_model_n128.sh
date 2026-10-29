#!/bin/bash
#
# Future Work B.1 — measure the same-model Jaccard at N=128 directly.
#
# Section 4.1 compares four Jaccard values; three are measured and one is not:
#   cross-seed  @N=32   0.741  measured
#   same model  @N=32   0.798  measured
#   cross-seed  @N=128  0.813  measured
#   same model  @N=128  0.889  ANALYTIC
# The training-variation component (0.076 = 0.187 - 0.111) is inferred from that
# analytic value. This run replaces it with a measurement.
#
# Evaluates the SAME checkpoint that produced the existing N=128 reference —
# seed 1, global_step_120 — a second time, independently, over all 1023 prompts.
#
# On the sampling seed. --seed is passed for a distinct output filename, but it
# does NOT drive sampling: gen_utils_sglang.py forwards only temperature, top_p,
# max_new_tokens and stop to the engine, and evaluate_model.py's seed_everything()
# seeds python/numpy/torch, none of which the sglang sampler consults. The two
# evaluations are therefore independent draws whatever seed is given — a fact
# already demonstrated at N=32, where two runs of one checkpoint returned
# Jaccard 0.798 rather than 1.000. Independence here is a property of the
# harness, not of the flag.
#
# Everything else matches the original reference exactly so the two are
# comparable: same checkpoint, temperature 0.7, max_new_tokens 1024, prompt-type
# qwen, 128 samples, all 1023 prompts.
#
# The model name avoids every glob used by the existing analysis scripts
# (qwen_0.5b_ckpt1_hi_*, qwen_0.5b_ckpt1_rep2_*, qwen_0.5b_run1_*), so nothing
# is overwritten and no downstream script silently absorbs these results.
#
# ~131k generations, ~2h.

set -euo pipefail

EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-192}"

CKPT_SEED=1
EVAL_SEED="${EVAL_SEED:-1}"        # differs from the reference run's 0
NAME="qwen_0.5b_ckpt${CKPT_SEED}_n128_rep2"
DATA="simplelr_qwen_level1to4_sub1k@train"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
LOG_DIR="$EXP/logs"; mkdir -p "$LOG_DIR"

cd "$REPO"

run_dir="$(ls -d "$REPO"/checkpoints/*_seed"${CKPT_SEED}" 2>/dev/null | head -1)"
HF="$run_dir/global_step_120/actor/huggingface"
[[ -d "$HF" ]] || { echo "no checkpoint at $HF" >&2; exit 1; }

busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
if [[ -n "$busy" ]]; then
    echo "GPU already in use by PID(s): ${busy//$'\n'/ }" >&2; exit 1
fi

pat="$REPO/results/${NAME}_${DATA}_*.json"
if compgen -G "$pat" >/dev/null; then
    echo "already done: $(basename "$(ls $pat | head -1)")"; exit 0
fi

echo "same-model repeat: seed ${CKPT_SEED} @ global_step_120, 1023 prompts x 128 samples, eval-seed ${EVAL_SEED}"
set +e
python -u inference/evaluate_model.py \
    --model-path "$HF" --tokenizer-path "$HF" \
    --test-data "$DATA" --num-samples 128 \
    --max-new-tokens 1024 --temperature 0.7 \
    --prompt-type qwen --save-path "$REPO/results" \
    --model-name "$NAME" --seed "$EVAL_SEED" \
    --gpu-memory-util "$GPU_MEM_UTIL" --use-sglang \
    > "$LOG_DIR/${NAME}.log" 2>&1
rc=$?
set -e
grep -aE "Finished Generating|pass@1" "$LOG_DIR/${NAME}.log" | tail -2 || true
if [[ $rc -ne 0 ]]; then
    echo "FAILED (exit $rc):" >&2; tail -15 "$LOG_DIR/${NAME}.log" >&2; exit "$rc"
fi
compgen -G "$pat" >/dev/null || { echo "exited 0 but wrote no results" >&2; exit 1; }
echo "done. next: python $EXP/same_model_n128.py"

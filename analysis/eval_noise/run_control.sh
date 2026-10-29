#!/bin/bash
#
# E0b -- eval-noise control. Waits for E0 to finish, then runs two evals on a
# SINGLE existing checkpoint. No training.
#
#   bash run_control.sh            wait for E0, then run
#   NOWAIT=1 bash run_control.sh   run immediately (GPU must be free)
#
# The question: E0 measured label instability across TRAINING seeds. How much of
# that is training variation, and how much is just pass@1 measurement noise?
#
#   A. Re-evaluate one checkpoint at 32 samples, independently sampled.
#      Same weights, same prompts, only the sampling differs. If the flagged-set
#      Jaccard here matches the ~0.72-0.76 seen ACROSS training seeds, then the
#      instability is measurement noise and more eval samples -- not more seeds --
#      is the correct fix.
#
#   B. Evaluate the same checkpoint at 128 samples on the near-threshold prompts,
#      to get a precise pass@1 per prompt. That pins down each prompt's true rate
#      and lets analyze_noise.py predict analytically how much label churn 32-sample
#      evaluation must produce, independent of any training difference.
#
# sglang is NOT seeded (only temperature/top_p reach the engine; seed_everything()
# seeds python/numpy/torch, which do not drive sglang sampling), so repeat runs are
# genuinely independent. --seed only distinguishes the output filenames, which is
# why this calls evaluate_model.py directly: inference/test.sh rejects unknown
# flags and never forwards --seed.

set -euo pipefail

EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
E0DIR="$(cd "$EXP/../E0_seed_stability" && pwd)"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-192}"

CKPT_SEED="${CKPT_SEED:-1}"        # which trained checkpoint to re-evaluate
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
MAX_LEN="${MAX_LEN:-1024}"
TEMP="${TEMP:-0.7}"
LOG_DIR="$EXP/logs"; mkdir -p "$LOG_DIR"

cd "$REPO"

run_dir="$(ls -d "$REPO"/checkpoints/*_seed"${CKPT_SEED}" 2>/dev/null | head -1)"
HF="$run_dir/global_step_120/actor/huggingface"
[[ -d "$HF" ]] || { echo "no checkpoint at $HF" >&2; exit 1; }

#-------------------------------------------------------------------
# Wait for E0 -- one GPU, so the control cannot overlap training
#-------------------------------------------------------------------
if [[ "${NOWAIT:-0}" != "1" ]]; then
    if pgrep -f 'run_all[.]sh' >/dev/null 2>&1; then
        echo "E0 still running; waiting for it to finish before touching the GPU..."
        while pgrep -f 'run_all[.]sh' >/dev/null 2>&1; do sleep 60; done
        echo "E0 finished. Starting control."
        sleep 30                       # let ray/sglang release VRAM
    fi
fi

evaluate() {   # <model_name> <test_data> <num_samples> <eval_seed>
    local name="$1" data="$2" n="$3" es="$4"
    local pat="$REPO/results/${name}_${data}_*.json"
    if compgen -G "$pat" >/dev/null; then
        echo "  already done: $(basename "$(ls $pat | head -1)")"; return
    fi
    echo "  -> $name  ($data, ${n} samples, eval-seed ${es})"
    # NOTE: do not wrap this pipeline in `|| true`. With pipefail that would
    # swallow a python crash and let the script march on to the next stage,
    # leaving the analysis to fail later with a confusing "missing eval".
    set +e
    python -u inference/evaluate_model.py \
        --model-path "$HF" --tokenizer-path "$HF" \
        --test-data "$data" --num-samples "$n" \
        --max-new-tokens "$MAX_LEN" --temperature "$TEMP" \
        --prompt-type qwen --save-path "$REPO/results" \
        --model-name "$name" --seed "$es" \
        --gpu-memory-util "$GPU_MEM_UTIL" --use-sglang \
        > "$LOG_DIR/${name}.log" 2>&1
    local rc=$?
    set -e
    grep -aE "Finished|pass@" "$LOG_DIR/${name}.log" | tail -5 || true
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED (exit $rc). Last lines of $LOG_DIR/${name}.log:" >&2
        tail -15 "$LOG_DIR/${name}.log" >&2
        exit "$rc"
    fi
    compgen -G "$pat" >/dev/null || {
        echo "  eval exited 0 but wrote no results matching $pat" >&2; exit 1; }
}

#-------------------------------------------------------------------
# A. Independent 32-sample repeat of the same checkpoint
#-------------------------------------------------------------------
echo "[A] repeat eval, 32 samples, same checkpoint (seed ${CKPT_SEED})"
evaluate "qwen_0.5b_ckpt${CKPT_SEED}_rep2" "simplelr_qwen_level1to4_sub1k@train" 32 1

#-------------------------------------------------------------------
# B. High-sample eval on the near-threshold subset
#-------------------------------------------------------------------
# Refresh E0's partition first: the subset is defined from per_example.csv, and
# after seed 5 lands that file is stale at K=4 unless the analysis is re-run.
echo "[B] refreshing E0 analysis so the subset reflects every finished seed"
bash "$E0DIR/classify.sh" >/dev/null 2>&1 || true
python "$E0DIR/analyze_stability.py" >/dev/null 2>&1 || true

echo "[B] building near-threshold subset"
python "$EXP/make_noise_subset.py"

echo "[B] high-sample eval, 128 samples"
evaluate "qwen_0.5b_ckpt${CKPT_SEED}_hi" "simplelr_qwen_level1to4_noise@train" 128 0

echo
echo "control done. next: python $EXP/analyze_noise.py"

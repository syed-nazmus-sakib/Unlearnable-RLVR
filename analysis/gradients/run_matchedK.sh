#!/bin/bash
#
# A12.2 matched-K test and A12.3 within-group attenuation curve.
#
#   bash run_matchedK.sh
#
# Registered plan:
#   K=8  primary,     3 random subsamples
#   K=4  sensitivity, 3 random subsamples
#   K=11 and K=6 for the attenuation curve (3 subsamples at 11, ONE at 6 --
#        curve only, spread not estimated there, and that is stated on the figure)
# Both K values are reported regardless of which way they fall (A12.2).
#
# The decisive number is easy-easy at K=11, D_u's natural median: if easy falls
# from ~0.33 toward D_u's ~0.12 simply by being given D_u's sample size, the
# artifact alone can produce the whole reported gap.
#
# Each (K, subsample) is: build a rollouts file with exactly K correct and K
# incorrect per prompt -> Chen's unmodified script in CHUNK-sized prompt batches
# (his loop holds every prompt's lm_head gradient in RAM; the natural-n run
# OOM-killed the machine twice before this was chunked) -> group cosine means
# into analysis/attenuation.json -> DELETE that run's .npz and .npy.
#
# The delete is not optional: one K-run's gradients are ~35 GB flattened and ten
# runs would need ~350 GB. Only the group means survive, which is all the panel
# needs.
set -uo pipefail
EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
PY="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}/python"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHUNK="${CHUNK:-8}"
RSS_LIMIT_GB="${RSS_LIMIT_GB:-16}"
MIN_FREE_GB="${MIN_FREE_GB:-3}"
RUNS="${RUNS:-4:3 6:1 8:3 11:3}"
mkdir -p "$EXP/logs_matchedK" "$EXP/analysis"

watchdog() {
    while kill -0 "$1" 2>/dev/null; do
        rss=$(awk '/VmRSS/{print int($2/1048576)}' /proc/"$1"/status 2>/dev/null || echo 0)
        free=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        if [[ "${rss:-0}" -gt "$RSS_LIMIT_GB" || "${free:-99}" -lt "$MIN_FREE_GB" ]]; then
            echo "  WATCHDOG: rss=${rss}G free=${free}G -> killing $1"; kill -9 "$1" 2>/dev/null; return
        fi
        sleep 20
    done
}

echo "############ matched-K starting $(date '+%F %H:%M') ############"
for spec in $RUNS; do
    K="${spec%%:*}"; NSUB="${spec##*:}"
    for SUB in $(seq 1 "$NSUB"); do
        DIR="$EXP/gradsim_grads_K${K}_s${SUB}"
        FLATD="$EXP/gradsim_flat_K${K}_s${SUB}"
        LOG="$EXP/logs_matchedK/K${K}_s${SUB}.log"
        echo "===== K=$K subsample $SUB  $(date '+%H:%M') ====="
        mkdir -p "$DIR"
        "$PY" "$EXP/build_matchedK.py" "$K" "$SUB" || { echo "  build failed"; continue; }
        ROLL="$REPO/results/e1_gradsim_K${K}_s${SUB}_GUARDED.json"

        prev=-1
        while :; do
            n=$(ls "$DIR"/average_gradients_prompt_correct_*.npz 2>/dev/null | wc -l)
            [[ "$n" -eq "$prev" ]] && break
            prev="$n"
            (cd "$REPO" && python -u gradient_analysis/checkpoint_gradient_analysis.py \
                --rollouts_file "$ROLL" \
                --model_path "$(cd "$REPO/.." && pwd)/models/Qwen2.5-0.5B" \
                --output_dir "$DIR" \
                --lora --lora_r 16 --lora_alpha 32 \
                --prompt_type qwen --simplified \
                --batch_size 4 --max_memory_gb 18 \
                --start_idx 0 --end_idx "$CHUNK" >> "$LOG" 2>&1) &
            py=$!; watchdog "$py" & wd=$!
            wait "$py"; kill "$wd" 2>/dev/null
            rm -f "$DIR"/average_gradients_prompt_[0-9]*.npz \
                  "$DIR"/average_gradients_prompt_incorrect_*.npz
        done
        echo "  gradients: $(ls "$DIR"/average_gradients_prompt_correct_*.npz 2>/dev/null | wc -l) prompts"

        "$PY" "$EXP/matchedK_similarity.py" "$DIR" "$K" "$SUB" 2>&1 | tail -5
        rm -rf "$DIR" "$FLATD" "$ROLL"
        echo "  cleaned up; disk avail $(df -BG --output=avail "$EXP" | tail -1 | tr -d ' ')"
    done
done
echo "############ matched-K done $(date '+%F %H:%M') ############"
"$PY" "$EXP/make_figure5.py" 2>&1 | tail -25

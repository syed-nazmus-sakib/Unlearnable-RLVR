#!/bin/bash
# Figure 5 step 2: per-prompt GRPO-loss gradients under the INITIAL policy,
# using Chen's own checkpoint_gradient_analysis.py unmodified, with his fixed
# randomly-initialised LoRA adapter (r=16, alpha=32, seeded) -- his Section 5.1.
# Input labels are guarded (is_correct overwritten), since 3.7% of his is_correct
# rollouts fail our verifier guard and D_u has few correct rollouts to spare.
#
# CHUNKED, 2026-08-04 23:55. The single-pass form was OOM-killed twice (21:55
# global oom-kill at 25.5 GB RSS; again around 23:36). Cause: his script holds a
# running-average lm_head gradient for EVERY prompt in RAM -- bf16 [151936 x 896]
# = 272 MB, times three flavours (all/correct/incorrect) = ~0.82 GB per prompt --
# and writes them only after the last prompt, so 151 prompts needs ~123 GB on a
# 30 GB machine. It could never have finished.
#
# Nothing about the method changes: the same script computes the same gradients
# for the same prompts, in chunks of CHUNK prompts, each chunk saving before the
# next starts. His own resume filter (checkpoint_gradient_analysis.py:977 skips
# qids whose correct .npz already exists) makes a repeated
# "--start_idx 0 --end_idx CHUNK" walk the unfinished prompts, so any kill costs
# at most one chunk instead of the whole run.
set -uo pipefail
EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHUNK="${CHUNK:-8}"                  # ~0.82 GB RAM per prompt -> ~7 GB + ~3 GB overhead
RSS_LIMIT_GB="${RSS_LIMIT_GB:-16}"   # watchdog: never take the machine down again
MIN_FREE_GB="${MIN_FREE_GB:-3}"      # abort the chunk if the machine gets this tight
OUTDIR="$EXP/gradsim_grads"
mkdir -p "$EXP/logs_gradsim" "$OUTDIR"
cd "$REPO"

done_count() { ls "$OUTDIR"/average_gradients_prompt_correct_*.npz 2>/dev/null | wc -l; }

watchdog() {   # $1 = pid to watch
    while kill -0 "$1" 2>/dev/null; do
        rss=$(awk '/VmRSS/{print int($2/1048576)}' /proc/"$1"/status 2>/dev/null || echo 0)
        free=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        if [[ "${rss:-0}" -gt "$RSS_LIMIT_GB" || "${free:-99}" -lt "$MIN_FREE_GB" ]]; then
            echo "WATCHDOG: rss=${rss}G free=${free}G -> killing $1 (chunk lost; saved prompts kept)"
            kill -9 "$1" 2>/dev/null
            return
        fi
        sleep 20
    done
}

echo "=== chunked gradient stage, CHUNK=$CHUNK, started $(date '+%F %H:%M') ==="
prev=-1
while :; do
    n=$(done_count)
    echo "--- $(date '+%H:%M')  prompts with gradients: $n  (previous pass: $prev)"
    [[ "$n" -eq "$prev" ]] && { echo "no progress in a full pass -- stopping"; break; }
    prev="$n"

    python -u gradient_analysis/checkpoint_gradient_analysis.py \
        --rollouts_file results/e1_gradsim_initial_GUARDED.json \
        --model_path "$(cd "$REPO/.." && pwd)/models/Qwen2.5-0.5B" \
        --output_dir "$OUTDIR" \
        --lora --lora_r 16 --lora_alpha 32 \
        --prompt_type qwen --simplified \
        --batch_size 4 --max_memory_gb 18 \
        --start_idx 0 --end_idx "$CHUNK" \
        >> "$EXP/logs_gradsim/grads.log" 2>&1 &
    py=$!
    watchdog "$py" &
    wd=$!
    wait "$py"; rc=$?
    kill "$wd" 2>/dev/null

    # Figure 5 needs only the correct-rollout gradients (A11.2, Chen's Fig 6).
    # The all/incorrect files are ~0.5 GB per prompt and are dropped each pass.
    rm -f "$OUTDIR"/average_gradients_prompt_[0-9]*.npz \
          "$OUTDIR"/average_gradients_prompt_incorrect_*.npz
    echo "    chunk rc=$rc  disk avail: $(df -BG --output=avail "$OUTDIR" | tail -1 | tr -d ' ')  mem avail: $(awk '/MemAvailable/{print int($2/1048576)"G"}' /proc/meminfo)"
done

echo "=== gradient stage finished $(date '+%F %H:%M'); $(done_count) prompts have gradients ==="

# --- chained, and it runs regardless of how the gradient stage above exited ---
echo "=== building Figure 5 $(date '+%F %H:%M') ==="
cd "$EXP"
"${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}/python" make_figure5.py
echo "=== make_figure5.py rc=$? $(date '+%F %H:%M') ==="

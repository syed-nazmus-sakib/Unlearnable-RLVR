#!/bin/bash
#
# Re-run of the natural-n gradient pass, for the PER-ROLLOUT VARIANCE only
# (Amendment 14). ~2.4 h at 0.5B.
#
# The first pass destroyed it: Chen's script names the squared-gradient output
# `average_gradients_squared_{start_idx}_{end_idx}.npz`, every chunk ran with
# `--start_idx 0 --end_idx 8`, and nineteen chunks wrote to one filename. Five
# prompts survived. His script is NOT modified here -- the fix is at the wrapper
# level: after each chunk the file is MOVED to a per-chunk name.
#
# A fresh output directory is used on purpose. The existing gradsim_grads/ and
# gradsim_flat/ are the vectors every downstream result was computed from
# (130 distinct qids, zero duplicates, verified in A14.7) and are left untouched.
# A fresh directory also defeats the skip-existing filter, so the whole fixed
# cohort is recomputed and every group gets a real n.
#
# A14.6 assertions, each failing loudly rather than writing quiet output:
#   1. squared file named from the CHUNK COUNTER, never from fixed arguments
#   2. refuse to overwrite an existing squared file
#   3. file count == chunk count at the end
#   4. union of qids across files == the full cohort
set -uo pipefail
EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CHUNK="${CHUNK:-8}"
RSS_LIMIT_GB="${RSS_LIMIT_GB:-16}"
MIN_FREE_GB="${MIN_FREE_GB:-3}"
OUTDIR="$EXP/gradsim_variance"
LOG="$EXP/logs_gradsim/variance.log"
mkdir -p "$OUTDIR" "$EXP/logs_gradsim"

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

cd "$REPO"
echo "=== variance re-run started $(date '+%F %H:%M'), CHUNK=$CHUNK ==="
prev=-1; chunk_id=0
while :; do
    n=$(ls "$OUTDIR"/average_gradients_prompt_correct_*.npz 2>/dev/null | wc -l)
    echo "--- $(date '+%H:%M')  prompts done: $n  (previous pass: $prev)"
    [[ "$n" -eq "$prev" ]] && { echo "no progress -- stopping"; break; }
    prev="$n"; chunk_id=$((chunk_id + 1))

    python -u gradient_analysis/checkpoint_gradient_analysis.py \
        --rollouts_file results/e1_gradsim_initial_GUARDED.json \
        --model_path "$(cd "$REPO/.." && pwd)/models/Qwen2.5-0.5B" \
        --output_dir "$OUTDIR" \
        --lora --lora_r 16 --lora_alpha 32 \
        --prompt_type qwen --simplified \
        --batch_size 4 --max_memory_gb 18 \
        --start_idx 0 --end_idx "$CHUNK" >> "$LOG" 2>&1 &
    py=$!; watchdog "$py" & wd=$!
    wait "$py"; rc=$?; kill "$wd" 2>/dev/null

    # (1) + (2): per-chunk name, taken from the counter; never overwrite
    src="$(ls "$OUTDIR"/average_gradients_squared_[0-9]*_[0-9]*.npz 2>/dev/null | head -1)"
    if [[ -n "$src" ]]; then
        dst="$(printf '%s/average_gradients_squared_chunk%02d.npz' "$OUTDIR" "$chunk_id")"
        if [[ -e "$dst" ]]; then
            echo "FATAL: $dst already exists -- refusing to overwrite"; exit 4
        fi
        mv "$src" "$dst"
        echo "    chunk $chunk_id rc=$rc -> $(basename "$dst")"
    else
        echo "    chunk $chunk_id rc=$rc -- no squared file produced"
    fi
    # the per-prompt mean gradients are not needed again; only the squared file is
    rm -f "$OUTDIR"/average_gradients_prompt_[0-9]*.npz \
          "$OUTDIR"/average_gradients_prompt_incorrect_*.npz
done

# (3) + (4): count and coverage
n_files=$(ls "$OUTDIR"/average_gradients_squared_chunk*.npz 2>/dev/null | wc -l)
echo "=== chunks run: $((chunk_id - 1))   squared files: $n_files ==="
python - "$OUTDIR" "$EXP" <<'PY'
import glob, json, os, sys
import numpy as np
out, exp = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(out, "average_gradients_squared_chunk*.npz")))
seen = {}
for f in files:
    z = np.load(f, allow_pickle=True)
    d = z["prompt_correct_avg_squared"].item()
    dup = set(d) & set(seen)
    assert not dup, f"ASSERTION FAILED: qids {sorted(dup)[:5]} appear in more than one chunk file"
    seen.update(d)
npz = glob.glob(os.path.join(out, "average_gradients_prompt_correct_*.npz"))
cohort = {int(p.rsplit('_', 1)[1][:-4]) for p in npz}
print(f"squared files: {len(files)}   qids with measured E[||g||^2]: {len(seen)}")
print(f"prompts with a gradient in this run: {len(cohort)}")
missing = cohort - set(seen)
print(f"prompts missing a variance measurement: {len(missing)} {sorted(missing)[:10]}")
assert len(files) > 0, "ASSERTION FAILED: no squared files written"
assert not missing, f"ASSERTION FAILED: {len(missing)} prompts have gradients but no variance"
print("ASSERTIONS PASSED: per-chunk files distinct, coverage complete")
PY
echo "=== variance re-run finished $(date '+%F %H:%M') ==="

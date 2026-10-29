#!/bin/bash
#
# E0 / one seed: GRPO train -> pass@1 eval -> prune -> record in manifest.
#
# Usage:  bash run_seed.sh <seed>
#
# Every stage is skip-if-done, so re-running after a crash resumes rather than
# repeating work. Three things here are not in repro/train_1gpu.sh and are the
# reason this wrapper exists:
#
#   1. Automatic stop at global_step_120. train_1gpu.sh runs 372 generation
#      batches, which overshoots to ~210 optimizer steps because the
#      gen-batch:opt-step ratio decays from 2.8 to 1.76 as the policy sharpens
#      (see ../../unlearnability-rlvr/repro/README.md). Seed 1 was stopped by
#      hand; at 4+ seeds that is both tedious and ~40% wasted GPU time. The
#      watchdog below polls for the step-120 checkpoint and terminates the run.
#
#   2. Pruning. Each run writes 3 saves x 8.5GB = 26GB, but E0 only ever reads
#      global_step_120/actor/huggingface (~1GB) and prompt_reward_stats.csv.
#      With 242GB free on this exFAT volume, 5 unpruned seeds would not fit.
#      Pruning happens only after the eval JSON exists, so a failed eval never
#      destroys the checkpoint it needs.
#
#   3. Process-group kill. train_1gpu.sh installs an EXIT/INT/TERM trap that
#      calls `ray stop`, but bash defers trap handlers while a foreground child
#      (`bash train.sh`) is running, so signalling that script alone does not
#      stop training promptly. `set -m` puts the job in its own process group
#      so the whole tree can be signalled at once; ray cleanup is then repeated
#      here because the deferred trap may never have fired.
#
# Env overrides: REPO_ROOT, VLLM_ENV_BIN, TARGET_STEP, PRUNE=0, MIN_FREE_GB.

set -euo pipefail

seed="${1:?Usage: $0 <seed>}"

EXP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$EXP_DIR/../../unlearnability-rlvr" && pwd)}"
TARGET_STEP="${TARGET_STEP:-120}"
PRUNE="${PRUNE:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"

MANIFEST="$EXP_DIR/manifest.tsv"
LOG_DIR="$EXP_DIR/logs"
MODEL_NAME="qwen_0.5b_seed${seed}"
TEST_DATA_TAG="simplelr_qwen_level1to4_sub1k@train"

export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

say() { printf '\n[E0 seed %s] %s\n' "$seed" "$*"; }

# checkpoints/<run>_seed<N> -- the glob is anchored at the end, so _seed1 does
# not also match _seed10.
find_run_dir() {
    local d
    d="$(ls -d "$REPO_ROOT"/checkpoints/*_seed"${seed}" 2>/dev/null | head -1 || true)"
    printf '%s' "$d"
}

hf_path_for() { printf '%s/global_step_%s/actor/huggingface' "$1" "$TARGET_STEP"; }

result_json() {
    ls "$REPO_ROOT"/results/"${MODEL_NAME}"_"${TEST_DATA_TAG}"*.json 2>/dev/null | head -1 || true
}

ray_cleanup() {
    ray stop --force >/dev/null 2>&1 || true
    pkill -f 'ray::' >/dev/null 2>&1 || true
    rm -rf /dev/shm/ray* /tmp/ray* >/dev/null 2>&1 || true
}

#-------------------------------------------------------------------
# Preflight
#-------------------------------------------------------------------
[[ -x "$(command -v python)" ]] || { echo "python not on PATH (VLLM_ENV_BIN wrong?)" >&2; exit 1; }
[[ -f "$REPO_ROOT/repro/train_1gpu.sh" ]] || { echo "no train_1gpu.sh under $REPO_ROOT" >&2; exit 1; }

free_gb="$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')"
if (( free_gb < MIN_FREE_GB )); then
    echo "only ${free_gb}GB free, need >=${MIN_FREE_GB}GB for one run (26GB before pruning)." >&2
    echo "Prune earlier seeds first:  PRUNE=1 bash run_seed.sh <old_seed>" >&2
    exit 1
fi
say "starting; ${free_gb}GB free"

#-------------------------------------------------------------------
# 1. Train (skip if the target checkpoint already exists)
#-------------------------------------------------------------------
run_dir="$(find_run_dir)"
if [[ -n "$run_dir" && -f "$(hf_path_for "$run_dir")/model.safetensors" ]]; then
    say "global_step_${TARGET_STEP} already present, skipping training"
else
    train_log="$LOG_DIR/train_seed${seed}.log"
    say "training -> $train_log  (~2.5h)"

    set -m                                  # own process group for the job
    bash "$REPO_ROOT/repro/train_1gpu.sh" "$seed" >"$train_log" 2>&1 </dev/null &
    pgid=$!
    set +m

    # Poll for a *complete* step-120 save: model.safetensors present and its
    # size unchanged across two consecutive polls, so we never stop mid-write.
    prev_size=-1
    while kill -0 "$pgid" 2>/dev/null; do
        run_dir="$(find_run_dir)"
        if [[ -n "$run_dir" ]]; then
            sft="$(hf_path_for "$run_dir")/model.safetensors"
            if [[ -f "$sft" ]]; then
                size="$(stat -c %s "$sft" 2>/dev/null || echo 0)"
                if [[ "$size" -gt 0 && "$size" == "$prev_size" ]]; then
                    say "step ${TARGET_STEP} checkpoint complete -> stopping training"
                    kill -TERM -"$pgid" 2>/dev/null || true
                    sleep 10
                    kill -KILL -"$pgid" 2>/dev/null || true
                    break
                fi
                prev_size="$size"
            fi
        fi
        sleep 30
    done
    wait "$pgid" 2>/dev/null || true
    ray_cleanup

    run_dir="$(find_run_dir)"
    [[ -n "$run_dir" && -f "$(hf_path_for "$run_dir")/model.safetensors" ]] || {
        echo "training finished without a step-${TARGET_STEP} checkpoint; see $train_log" >&2
        exit 1
    }
fi

[[ -f "$run_dir/prompt_reward_stats.csv" ]] || {
    echo "missing prompt_reward_stats.csv in $run_dir -- classification needs it" >&2
    exit 1
}

#-------------------------------------------------------------------
# 2. Evaluate pass@1 on the 1023 training prompts
#-------------------------------------------------------------------
res="$(result_json)"
if [[ -n "$res" ]]; then
    say "eval already done: ${res#$REPO_ROOT/}"
else
    say "evaluating (32 samples x 1023 prompts, ~30min)"
    bash repro/eval.sh "$(hf_path_for "$run_dir")" "$MODEL_NAME" \
        2>&1 | tee "$LOG_DIR/eval_seed${seed}.log"
    res="$(result_json)"
    [[ -n "$res" ]] || { echo "eval produced no results JSON" >&2; exit 1; }
fi

#-------------------------------------------------------------------
# 3. Prune -- only now that the eval JSON exists
#-------------------------------------------------------------------
if [[ "$PRUNE" == "1" ]]; then
    before="$(du -sh "$run_dir" 2>/dev/null | cut -f1)"
    for d in "$run_dir"/global_step_*; do
        [[ -d "$d" ]] || continue
        [[ "${d##*global_step_}" == "$TARGET_STEP" ]] || rm -rf "$d"
    done
    # FSDP shards + optimizer state; actor/huggingface/ is what eval and the
    # gradient analysis load, and is kept.
    rm -f "$run_dir/global_step_${TARGET_STEP}"/actor/*.pt \
          "$run_dir/global_step_${TARGET_STEP}"/data.pt 2>/dev/null || true
    say "pruned ${before} -> $(du -sh "$run_dir" 2>/dev/null | cut -f1)"
fi

#-------------------------------------------------------------------
# 4. Record in the manifest (repo-relative paths; classify.sh runs from REPO_ROOT)
#-------------------------------------------------------------------
[[ -f "$MANIFEST" ]] || printf 'seed\tmodel_name\trun_dir\tresults_glob\treward_stats\n' >"$MANIFEST"
# drop any previous row for this seed so re-runs update rather than duplicate
if grep -qP "^${seed}\t" "$MANIFEST"; then
    grep -vP "^${seed}\t" "$MANIFEST" >"$MANIFEST.tmp" || true
    mv "$MANIFEST.tmp" "$MANIFEST"
fi
printf '%s\t%s\t%s\t%s\t%s\n' \
    "$seed" "$MODEL_NAME" "${run_dir#$REPO_ROOT/}" \
    "results/${MODEL_NAME}_${TEST_DATA_TAG}*.json" \
    "${run_dir#$REPO_ROOT/}/prompt_reward_stats.csv" >>"$MANIFEST"

# keep manifest sorted by seed, header first
{ head -1 "$MANIFEST"; tail -n +2 "$MANIFEST" | sort -n; } >"$MANIFEST.tmp" && mv "$MANIFEST.tmp" "$MANIFEST"

say "done. manifest now has $(( $(wc -l <"$MANIFEST") - 1 )) seed(s)."

#!/bin/bash
#
# E3 / one (arm, seed): train -> eval on the two held-out sets -> prune -> delete.
#
#   bash run_arm_seed.sh <dataset_tag> <seed>
#
# Modelled on E0's run_seed.sh, with three differences that matter:
#
#   1. find_run_dir is anchored on the DATASET TAG as well as the seed. E0's
#      glob was checkpoints/*_seed<N>, which is unique when one dataset exists
#      and silently wrong here -- arm 2 seed 2 and arm 3 seed 2 both end in
#      _seed2, and `head -1` would have evaluated whichever sorted first.
#
#   2. Evaluation is on the HELD-OUT sets (A31.6), never on sub1k. Evaluating an
#      E3 arm on sub1k would measure training accuracy.
#
#   3. The checkpoint is DELETED after both evals succeed (A31.7). 20 runs of
#      2.4 GB would be 48 GB on an 85 GB volume; only the results JSONs are kept.
#
# Env: PRUNE=0 to keep checkpoints, TARGET_STEP, MIN_FREE_GB, KEEP=1 to skip the
# final delete.
set -uo pipefail

TAG="${1:?Usage: $0 <dataset_tag> <seed>}"
seed="${2:?Usage: $0 <dataset_tag> <seed>}"

EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
TARGET_STEP="${TARGET_STEP:-120}"
MIN_FREE_GB="${MIN_FREE_GB:-32}"
KEEP="${KEEP:-0}"
LOG_DIR="$EXP/logs"
MODEL_NAME="e3_${TAG}_seed${seed}"

export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export DATASET_NAME="simplelr_qwen_level1to4_${TAG}"
mkdir -p "$LOG_DIR"
cd "$REPO"

say() { printf '\n[E3 %s seed %s] %s\n' "$TAG" "$seed" "$*"; }

# anchored on BOTH the dataset name and the seed -- see note 1 above
find_run_dir() {
    ls -d "$REPO"/checkpoints/*"${DATASET_NAME}"*_seed"${seed}" 2>/dev/null | head -1 || true
}
hf_path_for() { printf '%s/global_step_%s/actor/huggingface' "$1" "$TARGET_STEP"; }
ray_cleanup() {
    ray stop --force >/dev/null 2>&1 || true
    pkill -f 'ray::' >/dev/null 2>&1 || true
    rm -rf /dev/shm/ray* /tmp/ray* >/dev/null 2>&1 || true
}
res_for() { ls "$REPO"/results/"${MODEL_NAME}"_"$1"*.json 2>/dev/null | head -1 || true; }

[[ -d "$REPO/data/SimpleRL-Zoo-Data/$DATASET_NAME" ]] || {
    echo "no dataset dir for $DATASET_NAME -- run build_arms.py" >&2; exit 2; }

# ---- CELL-LEVEL SKIP. This must come BEFORE the training check.
# The training check below asks "does a checkpoint exist?", but A31.7 deletes the
# checkpoint once both evals have written their JSON. A finished cell therefore
# looks unstarted, and a relaunch after the 2026-08-09 reboot began RETRAINING
# arm 4 seed 1 from scratch -- ~3.2 h of pure waste across seeds 1 and 2, and it
# would have overwritten results already on disk. The results JSONs are the only
# durable record of a completed cell, so they are what "done" is measured by.
if [[ -n "$(ls "$REPO"/results/"${MODEL_NAME}"_e3_math500*.json 2>/dev/null)" \
   && -n "$(ls "$REPO"/results/"${MODEL_NAME}"_e3_pool2000*.json 2>/dev/null)" ]]; then
    echo "[E3 $TAG seed $seed] both evals already on disk -- cell COMPLETE, skipping"
    exit 0
fi

free_gb="$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')"
(( free_gb >= MIN_FREE_GB )) || {
    echo "only ${free_gb}GB free, need >=${MIN_FREE_GB}GB (26GB before pruning)" >&2
    exit 3; }

# ------------------------------------------------------------------ 1. train
run_dir="$(find_run_dir)"
if [[ -n "$run_dir" && -f "$(hf_path_for "$run_dir")/model.safetensors" ]]; then
    say "step ${TARGET_STEP} already present, skipping training"
else
    tlog="$LOG_DIR/train_${TAG}_seed${seed}.log"
    say "training -> $tlog  (~1.6h, ${free_gb}GB free)"
    set -m
    bash "$REPO/repro/train_1gpu.sh" "$seed" >"$tlog" 2>&1 </dev/null &
    pgid=$!
    set +m
    prev=-1
    while kill -0 "$pgid" 2>/dev/null; do
        run_dir="$(find_run_dir)"
        if [[ -n "$run_dir" ]]; then
            sft="$(hf_path_for "$run_dir")/model.safetensors"
            if [[ -f "$sft" ]]; then
                sz="$(stat -c %s "$sft" 2>/dev/null || echo 0)"
                if [[ "$sz" -gt 0 && "$sz" == "$prev" ]]; then
                    say "step ${TARGET_STEP} complete -> stopping"
                    kill -TERM -"$pgid" 2>/dev/null || true; sleep 10
                    kill -KILL -"$pgid" 2>/dev/null || true; break
                fi
                prev="$sz"
            fi
        fi
        sleep 30
    done
    wait "$pgid" 2>/dev/null || true
    ray_cleanup
    run_dir="$(find_run_dir)"
    [[ -n "$run_dir" && -f "$(hf_path_for "$run_dir")/model.safetensors" ]] || {
        echo "no step-${TARGET_STEP} checkpoint; see $tlog" >&2; exit 4; }
fi

# ------------------------------------------------------------------ 2. eval
HF="$(hf_path_for "$run_dir")"
for SPEC in "e3_math500@test" "e3_pool2000@test"; do
    tagname="${SPEC%@*}"
    if [[ -n "$(res_for "$tagname")" ]]; then
        say "eval $tagname already done"; continue
    fi
    say "evaluating $SPEC (N=32)"
    TEST_DATA="$SPEC" NUM_SAMPLES=32 bash repro/eval.sh "$HF" "$MODEL_NAME" \
        >"$LOG_DIR/eval_${TAG}_seed${seed}_${tagname}.log" 2>&1
    [[ -n "$(res_for "$tagname")" ]] || {
        echo "eval $SPEC produced no results JSON; checkpoint KEPT for retry" >&2
        exit 5; }
done

# --------------------------------------------------- 3. prune, then delete
# Only after BOTH evals have written their JSON. A failed eval leaves the
# checkpoint in place so the run can be resumed rather than repeated.
if [[ "$KEEP" == "1" ]]; then
    say "KEEP=1, leaving $(du -sh "$run_dir" | cut -f1) in place"
else
    before="$(du -sh "$run_dir" 2>/dev/null | cut -f1)"
    rm -rf "$run_dir"
    say "deleted checkpoint ($before); results JSONs retained"
fi
say "done; $(df -BG --output=avail "$REPO" | tail -1 | tr -d ' ') free"

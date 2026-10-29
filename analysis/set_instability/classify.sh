#!/bin/bash
#
# E0 / classification: run the repo's finalize_unlearnable_example_id.py over
# every seed in manifest.tsv (or an explicit subset).
#
# Usage:
#   bash classify.sh              # all seeds in the manifest
#   bash classify.sh 1 2 3        # just these -- e.g. to reproduce the paper's
#                                 # 3-seed protocol, or to compare K=1 vs K=5
#
# Writes partitions/K<n>_seeds<list>/qwen_0.5b_math_level1to4_{unlearnable,
# learnable,no_reward,easy}.json. Each subset gets its own directory so the
# K=1 partition you already have stays intact for comparison.
#
# Definitions are the repo's, unchanged (threshold tau=0.1):
#   no_reward   = union over runs of prompts with max mean_reward == 0
#   unlearnable = intersection over runs of pass@1<=tau, minus no_reward
#   learnable   = initially flagged, minus the union of post-training flags
# Intersection is why seed count matters: each added seed can only shrink
# `unlearnable`, and how fast it shrinks is exactly what E0 measures.

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$EXP_DIR/../../unlearnability-rlvr" && pwd)}"
MANIFEST="$EXP_DIR/manifest.tsv"
THRESHOLD="${THRESHOLD:-0.1}"
INITIAL_GLOB="${INITIAL_GLOB:-results/qwen_0.5b_initial_*@train*.json}"

export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"

[[ -f "$MANIFEST" ]] || { echo "no manifest at $MANIFEST -- run run_seed.sh first" >&2; exit 1; }
cd "$REPO_ROOT"

want=("$@")
args=()
used=()

while IFS=$'\t' read -r seed model run_dir results reward; do
    [[ "$seed" == "seed" ]] && continue          # header
    [[ -z "$seed" ]] && continue
    if [[ ${#want[@]} -gt 0 ]]; then
        keep=0
        for w in "${want[@]}"; do [[ "$w" == "$seed" ]] && keep=1; done
        [[ $keep == 1 ]] || continue
    fi
    [[ -f "$REPO_ROOT/$reward" ]] || { echo "missing reward stats for seed $seed: $reward" >&2; exit 1; }
    args+=(--run "seed${seed}=${results}" --reward-stats "seed${seed}=${reward}")
    used+=("$seed")
done < "$MANIFEST"

[[ ${#used[@]} -gt 0 ]] || { echo "no seeds selected" >&2; exit 1; }

joined="$(IFS=-; echo "${used[*]}")"
OUT_DIR="$EXP_DIR/partitions/K${#used[@]}_seeds${joined}"
mkdir -p "$OUT_DIR"

echo "classifying over seeds: ${used[*]}  (tau=$THRESHOLD)"
python classification/finalize_unlearnable_example_id.py \
    "${args[@]}" \
    --initial-run "initial=${INITIAL_GLOB}" \
    --threshold "$THRESHOLD" \
    --output-dir "$OUT_DIR" \
    --output-prefix qwen_0.5b_math_level1to4 \
    2>&1 | tee "$OUT_DIR/classify.log"

echo
echo "partitions -> ${OUT_DIR#$EXP_DIR/}"
echo "next: python $EXP_DIR/analyze_stability.py"

#!/bin/bash
#
# E0 / all seeds: run_seed.sh over a list of seeds, sequentially.
#
# Usage:
#   bash run_all.sh 2 3 4 5          # explicit seeds
#   bash run_all.sh                  # default: 2 3 4 5 (seed 1 already done)
#
# Sequential by design -- one 24GB card, and each run colocates the sglang
# engine with the FSDP actor. Budget ~3h per seed (2.5h train + 0.5h eval),
# so the default list is roughly one overnight run.
#
# Long runs: nohup bash run_all.sh 2 3 4 5 > logs/run_all.log 2>&1 &
#
# A failing seed does not abort the rest; failures are collected and reported
# at the end, since a single OOM should not cost you the other three runs.

set -uo pipefail

EXP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
seeds=("$@")
[[ ${#seeds[@]} -gt 0 ]] || seeds=(2 3 4 5)

mkdir -p "$EXP_DIR/logs"
failed=()

echo "E0: seeds ${seeds[*]}  (~3h each)"
start_all=$SECONDS

for s in "${seeds[@]}"; do
    echo
    echo "================ seed $s ================"
    start=$SECONDS
    if bash "$EXP_DIR/run_seed.sh" "$s"; then
        printf 'seed %s ok in %dh%02dm\n' "$s" $(( (SECONDS-start)/3600 )) $(( ((SECONDS-start)%3600)/60 ))
    else
        echo "seed $s FAILED (see logs/train_seed${s}.log, logs/eval_seed${s}.log)" >&2
        failed+=("$s")
    fi
done

printf '\n================ summary ================\n'
printf 'elapsed: %dh%02dm\n' $(( (SECONDS-start_all)/3600 )) $(( ((SECONDS-start_all)%3600)/60 ))
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "failed seeds: ${failed[*]}"
    echo "re-run one with: bash run_seed.sh <seed>   (it resumes, it does not restart)"
    exit 1
fi
echo "all seeds done. next:  bash classify.sh && python analyze_stability.py"

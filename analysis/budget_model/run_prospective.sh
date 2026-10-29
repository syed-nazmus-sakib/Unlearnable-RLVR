#!/bin/bash
#
# A66.2 — PROSPECTIVE validation of the A65 budget rule.
#
# Four passes, 1023 prompts, cap 5120, trained ckpt1 (seed 1, global_step_120),
# temperature 0.7, prompt-type qwen -- identical to the A49 5120 run in every
# respect except --num-samples and the evaluation seed.
#
#   N=40  pass 1 (seed 0)   N=40  pass 2 (seed 1)
#   N=64  pass 1 (seed 0)   N=64  pass 2 (seed 1)
#
# The predictions were frozen and timestamped at 2026-08-15T23:47:23+06:00,
# BEFORE this ran: N=40 -> 0.8131 [0.7699, 0.8533], N=64 -> 0.8397
# [0.7980, 0.8792]. They are in analysis/budget_prospective_predictions.json
# and in A66.2. Nothing here can change them.
#
# N=40 first: it is the budget A65.5 says stably meets J >= 0.80, so it is the
# one a reader would adopt, and a prefix of this run that stops after two cells
# still answers the primary question.
set -uo pipefail

EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="$(cd "$EXP/../../unlearnability-rlvr" && pwd)"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-192}"

DATA="simplelr_qwen_level1to4_sub1k@train"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
STATUS="$EXP/analysis/prospective_status.jsonl"
FLOOR_GB="${FLOOR_GB:-5}"
cd "$REPO"

run_dir="$(ls -d "$REPO"/checkpoints/*_seed1 2>/dev/null | head -1)"
HF="$run_dir/global_step_120/actor/huggingface"
[[ -d "$HF" ]] || { echo "no checkpoint at $HF" >&2; exit 1; }

busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
[[ -z "$busy" ]] || { echo "GPU in use by PID(s): ${busy//$'\n'/ }" >&2; exit 1; }

say(){ printf '{"time":"%s","state":%s}\n' "$(date --iso-8601=seconds)" "$1" >> "$STATUS"; }
say '"start_prospective"'
echo "######## A66.2 prospective run starting $(date '+%F %H:%M') ########"
echo "checkpoint: $HF"

pass(){
    # NOTE: separate `local` statements. Bash expands the whole command line
    # BEFORE performing any of the assignments, so `local A=$1 B=$((A+1))` sees
    # an unset A and dies under `set -u`.
    local N="$1"
    local SEED="$2"
    local NAME="qwen_0.5b_ckpt1_n${N}_cap5120_p$((SEED+1))"
    echo "=== N=$N pass $((SEED+1)) (eval seed $SEED)  $(date '+%F %H:%M') ==="
    # A45 open_new: never overwrite an existing output.
    if compgen -G "$REPO/results/${NAME}_${DATA}_*.json" >/dev/null; then
        echo "$NAME: output exists, skipping (already collected)"; return 0
    fi
    local free_gb
    free_gb="$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9)"
    (( free_gb >= FLOOR_GB )) || {
        echo "only ${free_gb} GB free, refusing to start $NAME" >&2; return 1; }
    python -u inference/evaluate_model.py \
        --model-path "$HF" --tokenizer-path "$HF" \
        --test-data "$DATA" --num-samples "$N" \
        --max-new-tokens 5120 --temperature 0.7 \
        --prompt-type qwen --save-path "$REPO/results" \
        --model-name "$NAME" --seed "$SEED" \
        --gpu-memory-util "$GPU_MEM_UTIL" --use-sglang
    local rc=$?
    if (( rc == 0 )); then say "\"pass_done\",\"N\":$N,\"seed\":$SEED"
    else say "\"pass_failed\",\"N\":$N,\"seed\":$SEED,\"rc\":$rc"; fi
    echo "  rc=$rc  free now $(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9) GB"
    return $rc
}

for cell in "40 0" "40 1" "64 0" "64 1"; do
    set -- $cell
    pass "$1" "$2" || echo "cell N=$1 seed=$2 FAILED (continuing)" >&2
done

say '"all_done"'
echo "######## A66.2 prospective run done $(date '+%F %H:%M') ########"

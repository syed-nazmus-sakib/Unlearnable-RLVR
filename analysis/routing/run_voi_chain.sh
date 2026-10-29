#!/bin/bash
set -uo pipefail
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO=/media/rmedu/3039-8E90/LLM/unlearnability-rlvr
CKPT_ROOT=/home/rmedu/pr_b3_ckpts

run_one() {
    local ARM=$1 SEED=$2
    local TAG="pr_${ARM}_seed${SEED}"
    echo "=== START $TAG $(date -Is) ==="
    bash "$HERE/pr_train.sh" "$ARM" "$SEED" > "$CKPT_ROOT/${TAG}_driver.log" 2>&1
    echo "=== train rc=$? $TAG $(date -Is) ==="
    local HF_REAL=$( { compgen -G "$CKPT_ROOT/$TAG/global_step_120/actor/huggingface" || true; } | head -1)
    [ -z "$HF_REAL" ] && { echo "=== NO CKPT $TAG ==="; return; }
    ln -sfn "$HF_REAL" "$CKPT_ROOT/${TAG}_Qwen2.5-0.5B"
    cd "$REPO"; export RESULTS_DIR="$REPO/results"; local ok_all=1
    for SPEC in e3_math500 e3_pool2000; do
        TEST_DATA=${SPEC}@test NUM_SAMPLES=32 MAX_LENGTH=1024 GPU_MEM_UTIL=0.60 \
          bash repro/eval.sh "$CKPT_ROOT/${TAG}_Qwen2.5-0.5B" "b3_${ARM}_seed${SEED}" \
          > "$CKPT_ROOT/${TAG}_eval_${SPEC}.log" 2>&1
        echo "=== EVAL $SPEC rc=$? $TAG $(date -Is) ==="
        ls "$REPO/results"/b3_${ARM}_seed${SEED}_${SPEC}@test*.json >/dev/null 2>&1 || ok_all=0
    done
    [ "$ok_all" = "1" ] && rm -rf "$CKPT_ROOT/$TAG" && rm -f "$CKPT_ROOT/${TAG}_Qwen2.5-0.5B" \
      && echo "=== cleaned $TAG ===" || echo "=== KEPT $TAG ==="
}

for combo in "voi 1" "vanilla 2" "voi 2"; do run_one $combo; done
echo "=== VOI CHAIN ALL_DONE $(date -Is) ==="

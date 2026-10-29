#!/bin/bash
# B3 pilot overnight chain v2 (A83): train -> eval x2 -> delete-if-evals-exist.
# Fixes vs v1: absolute pr_train.sh path; eval goes through a symlink whose
# name contains the size token sglang's dispatcher requires; checkpoints are
# only deleted when BOTH results JSONs exist.
set -uo pipefail
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO=/media/rmedu/3039-8E90/LLM/unlearnability-rlvr
CKPT_ROOT=/home/rmedu/pr_b3_ckpts

for ARM in vanilla cert09; do
    TAG="pr_${ARM}_seed1"
    echo "=== CHAIN START arm=$ARM $(date -Is) ==="
    bash "$HERE/pr_train.sh" "$ARM" 1 > "$CKPT_ROOT/${TAG}_driver.log" 2>&1
    echo "=== train rc=$? arm=$ARM $(date -Is) ==="

    HF_REAL=$( { compgen -G "$CKPT_ROOT/$TAG/global_step_120/actor/huggingface" || true; } | head -1)
    if [ -z "$HF_REAL" ] || [ ! -d "$HF_REAL" ]; then
        echo "=== EVAL SKIPPED: no step-120 ckpt for $ARM ==="
        continue
    fi
    LINK="$CKPT_ROOT/${TAG}_Qwen2.5-0.5B"
    ln -sfn "$HF_REAL" "$LINK"

    cd "$REPO"
    export RESULTS_DIR="$REPO/results"
    ok_all=1
    for SPEC in e3_math500 e3_pool2000; do
        echo "=== EVAL $SPEC arm=$ARM start $(date -Is) ==="
        TEST_DATA=${SPEC}@test NUM_SAMPLES=32 MAX_LENGTH=1024 \
            GPU_MEM_UTIL=0.60 bash repro/eval.sh "$LINK" "b3_${ARM}_seed1" \
            > "$CKPT_ROOT/${TAG}_eval_${SPEC}.log" 2>&1
        rc=$?
        echo "=== EVAL $SPEC rc=$rc $(date -Is) ==="
        ls "$REPO/results"/b3_${ARM}_seed1_${SPEC}@test*.json >/dev/null 2>&1 || ok_all=0
    done

    if [ "$ok_all" = "1" ]; then
        rm -f "$LINK"
        rm -rf "$CKPT_ROOT/$TAG"
        echo "=== deleted ckpt dir for $ARM (both result JSONs verified) ==="
    else
        rm -f "$LINK"
        echo "=== KEPT ckpt dir for $ARM (an eval failed) ==="
    fi
done
echo "=== CHAIN ALL_DONE $(date -Is) ==="

#!/bin/bash
#
# E3 driver — arm 1 evals (already trained in E0), then arms 2-5 x 5 seeds.
#
#   setsid nohup ./run_all_e3.sh </dev/null >> logs_e3.out 2>&1 &
#
# Order is deliberate. Arm 1 costs no training, so its five evals land first and
# establish the baseline; then arms 4 and 5 -- the POSITIVE CONTROLS -- run
# before arms 2 and 3. A31.2 makes the whole experiment uninterpretable if the
# controls do not separate, so it is worth learning that after ~13 h rather than
# after ~33 h.
set -uo pipefail
EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
SEEDS="${SEEDS:-1 2 3 4 5}"
ARMS="${ARMS:-e3arm4_rand404 e3arm5_rand218 e3arm2_union e3arm3_inter}"
mkdir -p "$EXP/logs"

echo "############ E3 starting $(date '+%F %H:%M') ############"
echo "arms: $ARMS"
echo "seeds: $SEEDS"

# ---- arm 1: evaluate the existing E0 baseline checkpoints, no training ----
for s in $SEEDS; do
    d="$(ls -d "$REPO"/checkpoints/*sub1k*_seed"${s}" 2>/dev/null | head -1)"
    HF="$d/global_step_120/actor/huggingface"
    MODEL="e3_arm1_base_seed${s}"
    [[ -f "$HF/model.safetensors" ]] || { echo "arm1 seed $s: no checkpoint, SKIPPED"; continue; }
    for SPEC in e3_math500@test e3_pool2000@test; do
        tag="${SPEC%@*}"
        if ls "$REPO"/results/"${MODEL}"_"${tag}"*.json >/dev/null 2>&1; then
            echo "--- arm1 seed $s $tag already done"; continue
        fi
        echo "--- arm1 seed $s $SPEC  $(date '+%H:%M')"
        (cd "$REPO" && TEST_DATA="$SPEC" NUM_SAMPLES=32 \
            bash repro/eval.sh "$HF" "$MODEL" \
            >"$EXP/logs/eval_arm1_seed${s}_${tag}.log" 2>&1)
        echo "    rc=$?"
    done
done

# ---- arms 2-5 ----
for arm in $ARMS; do
    for s in $SEEDS; do
        echo "===== $arm seed $s  $(date '+%F %H:%M') ====="
        bash "$EXP/run_arm_seed.sh" "$arm" "$s"
        rc=$?
        echo "    rc=$rc  free $(df -BG --output=avail "$REPO" | tail -1 | tr -d ' ')"
        if [[ $rc -ne 0 ]]; then
            echo "    NON-ZERO EXIT -- continuing to the next seed; this cell will be"
            echo "    missing from the analysis and must be reported as such, not"
            echo "    silently dropped (A18.5)."
        fi
    done
done

echo "############ E3 training+eval done $(date '+%F %H:%M') ############"

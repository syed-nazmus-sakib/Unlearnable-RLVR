#!/bin/bash
set -euo pipefail
EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="$(cd "$EXP/../../unlearnability-rlvr" && pwd)"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-192}"

run_dir="$(ls -d "$REPO"/checkpoints/*_seed1 2>/dev/null | head -1)"
HF="$run_dir/global_step_120/actor/huggingface"
RAW="$REPO/results/n242_topup_p2_raw_402.json"
FINAL="$REPO/results/n242_topup_p2_simplelr_qwen_level1to4_sub1k@train_0.7_178_1024_None_402_qwen.json"

echo "=== Stage 1: Generate n242_topup_p2 (N=178, seed=402) and save raw JSON ==="
python -u "$EXP/run_n242_p2_safe.py" --stage gen --model-path "$HF" --raw-out "$RAW" --final-out "$FINAL" --num-samples 178 --seed 402

echo "=== Stage 2: Verify raw JSON in clean CPU process (20k chunks) ==="
python -u "$EXP/run_n242_p2_safe.py" --stage verify --model-path "$HF" --raw-out "$RAW" --final-out "$FINAL" --num-samples 178 --seed 402

echo "=== Stage 3: Compute N=242 guarded Jaccard vs frozen prediction ==="
python -u "$EXP/measure_prospective_n242.py"
python -u "$EXP/make_figure_9.py"
cp "$EXP/figures_pub/figure_9.png" "$EXP/../../camera_ready/experimental_results_bundle/figure9.png"
cp "$EXP/figures_pub/figure_9.png" "$EXP/../../camera_ready/NeurIPS_fig/figure9.png"
rm -f "$RAW"
echo "=== ALL DONE $(date -Is) ==="

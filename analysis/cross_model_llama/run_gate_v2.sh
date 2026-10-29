#!/bin/bash
# A34.2 gate: 800 MATH-Hard prompts x 32 samples, Llama-3.2-3B-Instruct.
# Generation only -- no gradients, no training. ~3.6 h at the projected rate.
set -uo pipefail
EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="$(cd "$EXP/../../unlearnability-rlvr" && pwd)"
MODEL="$(cd "$EXP/../../models/Llama-3.2-3B-Instruct" && pwd)"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
cd "$REPO"
echo "######## E4 gate v2 (A39.3 attempt 2) starting $(date '+%F %H:%M') ########"
# prompt_type=qwen is a MISNOMER in this harness: it selects the wrapper
# template only, and the chat template comes from the tokenizer at
# --tokenizer-path, i.e. Llama's own. Using the same wrapper as the 0.5B runs
# keeps the instruction identical across models (A34.6).
TEST_DATA="e4_screen800_v2@test" NUM_SAMPLES=32 MAX_LENGTH=1024 \
  GPU_MEM_UTIL=0.70 bash repro/eval.sh "$MODEL" e4_gate_llama3b_v2 --prompt-type e4v2
echo "rc=$?"
echo "######## E4 gate v2 (A39.3 attempt 2) done $(date '+%F %H:%M') ########"

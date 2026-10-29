#!/bin/bash
# E5 gate: Qwen2.5-1.5B base, the SAME 1023 prompts as the 0.5B work, N=32.
# Generation only. Registered relative gate in A48; nothing proceeds without it.
# MAX_LENGTH is held at 1024 to match the 0.5B reference it is compared against
# (A47: that cap is our own, and differs from Chen's 5120 default -- the
# comparison is like-for-like only because BOTH sides carry it).
set -uo pipefail
EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="$(cd "$EXP/../../unlearnability-rlvr" && pwd)"
MODEL="$(cd "$EXP/../../models/Qwen2.5-1.5B" && pwd)"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
cd "$REPO"
echo "######## E5 gate (Qwen2.5-1.5B) starting $(date '+%F %H:%M') ########"
TEST_DATA="simplelr_qwen_level1to4_sub1k@train" NUM_SAMPLES=32 MAX_LENGTH=1024 \
  GPU_MEM_UTIL=0.70 bash repro/eval.sh "$MODEL" e5_gate_qwen15b
echo "rc=$?"
echo "######## E5 gate done $(date '+%F %H:%M') ########"

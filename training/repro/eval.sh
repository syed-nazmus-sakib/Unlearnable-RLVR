#!/bin/bash
#
# pass@k evaluation wrapper for this box (1x RTX 3090 Ti, 24GB).
#
# Usage:
#   bash repro/eval.sh <model_path> <model_name> [extra args for inference/test.sh]
#
# Why this wrapper exists rather than calling inference/test.sh directly:
#
#   1. --test-data uses the *train* split. The README's step 1b evaluates
#      `simplelr_qwen_level1to4@test`, but classification/finalize_unlearnable_example_id.py
#      joins eval `question_id` against training `prompt_id`, and both are
#      extra_info["index"] -- 0..8138 on train, 0..499 on test for *different*
#      problems. Evaluating @test silently scores unrelated prompts against the
#      reward stats. The unlearnability question is about training prompts anyway.
#
#   2. SGLANG_MAX_RUNNING_REQUESTS caps sglang concurrency. gen_utils_sglang.py
#      submits 5000 prompts per generate() call; with Qwen2.5's 151936-token vocab
#      the logits tensor is ~0.6MB fp32 per running sequence, so an unbounded
#      decode batch OOMs 24GB long before the KV pool does.
#
#   3. --gpu-memory-util is passed through and now actually reaches sglang
#      (mem_fraction_static). It was previously dropped on the sglang path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO_ROOT"

model_path="${1:?Usage: $0 <model_path> <model_name> [extra args]}"
model_name="${2:?Usage: $0 <model_path> <model_name> [extra args]}"
shift 2

export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export DATA_ROOT="${DATA_ROOT:-./data}"
export RESULTS_DIR="${RESULTS_DIR:-./results}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-192}"

TEST_DATA="${TEST_DATA:-simplelr_qwen_level1to4_sub1k@train}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"

bash inference/test.sh "$model_path" \
    --model-name "$model_name" \
    --test-data "$TEST_DATA" \
    --num-samples "$NUM_SAMPLES" \
    --max-length "$MAX_LENGTH" \
    --gpu-memory-util "$GPU_MEM_UTIL" \
    "$@"

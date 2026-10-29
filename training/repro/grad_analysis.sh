#!/bin/bash
#
# Gradient analysis (root README step 2), rescaled for 1x RTX 3090 Ti.
#
# Usage:
#   bash repro/grad_analysis.sh
#
# Knobs (env vars, with the paper's values in parens):
#   N_PER_GROUP     prompts sampled per group        default 40   (paper: 100)
#   N_ROLLOUTS      rollouts per sampled prompt      default 200  (paper: 1000)
#   EXAMPLE_IDS_DIR                                  default ./example_ids
#   IDS_PREFIX                                       default qwen_0.5b_math_level1to4
#   RUN_TAG         subdir under grpo_gradient_analysis_output  default qwen_0.5b_1gpu
#
# Why smaller: 300 prompts x 1000 rollouts = 300k generations. At the throughput
# this box gets for Qwen2.5-0.5B that is many hours; 40 x 200 = 8k generations per
# group keeps the same block structure (easy / learnable / unlearnable) visible.
# Per-prompt gradients are averaged over *correct* rollouts only (--pos_only), so
# N_ROLLOUTS mainly controls how many correct samples each average is built from.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"
export DATA_ROOT="${DATA_ROOT:-./data}"
export RESULTS_DIR="${RESULTS_DIR:-./results}"
export GRAD_ANALYSIS_OUTPUT_DIR="${GRAD_ANALYSIS_OUTPUT_DIR:-./grpo_gradient_analysis_output}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-192}"

N_PER_GROUP="${N_PER_GROUP:-40}"
N_ROLLOUTS="${N_ROLLOUTS:-200}"
EXAMPLE_IDS_DIR="${EXAMPLE_IDS_DIR:-./example_ids}"
IDS_PREFIX="${IDS_PREFIX:-qwen_0.5b_math_level1to4}"
RUN_TAG="${RUN_TAG:-qwen_0.5b_1gpu}"
MODEL_PATH="${MODEL_PATH:-$(cd "$REPO_ROOT/.." && pwd)/models/Qwen2.5-0.5B}"
GRAD_GROUPS="${GRAD_GROUPS:-easy learnable unlearnable}"

# The source JSONL is written by the first eval run over the dataset. This repro
# evaluates the *train* split (see repro/README.md), so it is train_parsed.jsonl.
SOURCE_JSONL="${SOURCE_JSONL:-./data/simplelr_qwen_level1to4_sub1k/train_parsed.jsonl}"

echo "=== (a) sample prompts from each group ==="
python gradient_analysis/sample_grad_sim_prompts.py \
    --example-ids-dir    "$EXAMPLE_IDS_DIR" \
    --example-ids-prefix "$IDS_PREFIX" \
    --groups             $GRAD_GROUPS \
    --n-per-group        "$N_PER_GROUP" \
    --source-test-jsonl  "$SOURCE_JSONL" \
    --output-jsonl       ./data/grad_sim/test.jsonl

echo "=== (b) rollouts at the initial checkpoint ==="
# Note: inference/test.sh hardcodes temperature=0.7, which is what the paper's
# gradient-analysis rollouts use, so test.sh is fine here.
# NUM_SAMPLES must be set from N_ROLLOUTS, else eval.sh falls back to its own
# default of 32 and the per-prompt gradients average over far too few correct rollouts.
NUM_SAMPLES="$N_ROLLOUTS" TEST_DATA=./data/grad_sim/test.jsonl \
bash repro/eval.sh "$MODEL_PATH" "qwen_0.5b_initial_grad_sim"

echo "=== (c) per-prompt GRPO gradients (correct rollouts only, LoRA) ==="
n_prompts=$(( $(echo "$GRAD_GROUPS" | wc -w) * N_PER_GROUP ))
ROLLOUTS_FILE="./results/qwen_0.5b_initial_grad_sim_*.json" \
MODEL_PATH="$MODEL_PATH" \
OUTPUT_DIR="$GRAD_ANALYSIS_OUTPUT_DIR/$RUN_TAG" \
bash gradient_analysis/run_gradient.sh 0 "$n_prompts" --pos_only --lora

echo "=== (d) pairwise cosine similarity ==="
bash gradient_analysis/cal_similarity.sh "$RUN_TAG" 0 100000

echo "=== (e) block-ordered heatmap ==="
mkdir -p figures
python gradient_analysis/plot_grad_sim_heatmap.py \
    --similarity-files   "$GRAD_ANALYSIS_OUTPUT_DIR/$RUN_TAG/qid_cosine_similarity_pair_dict_pos_*.json" \
    --example-ids-dir    "$EXAMPLE_IDS_DIR" \
    --example-ids-prefix "$IDS_PREFIX" \
    --groups             $GRAD_GROUPS \
    --output             figures/grad_sim_heatmap.png

echo "Done -> figures/grad_sim_heatmap.png"

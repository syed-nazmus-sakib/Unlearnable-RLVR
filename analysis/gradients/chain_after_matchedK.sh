#!/bin/bash
# After run_matchedK.sh finishes: re-run K=4 and K=6 to recover their PER-PROMPT
# similarities, then rebuild the report and Figure 5.
#
# Why the re-run is needed: matchedK_similarity.py originally stored the group
# MEAN only, and run_matchedK.sh deletes each run's gradients immediately (ten
# runs would need ~350 GB). When cohort drift was noticed -- raising K drops
# exactly the prompts with fewest correct rollouts, so the K=11 cohort is the
# EASIEST 28 D_u prompts rather than the same D_u -- the fix required per-prompt
# values at every K, and K=4/K=6 no longer had them. K=8 and K=11 were caught in
# time; these two are re-measured.
#
# The re-run is exact, not approximate: build_matchedK.py seeds its sampler with
# 1000*K + subsample, so the same rollout files are regenerated bit-identically.
set -uo pipefail
EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PY="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}/python"
WATCH="${MATCHEDK_PID:?set MATCHEDK_PID}"
echo "=== armed $(date '+%F %H:%M'), waiting on matched-K pid $WATCH ==="
while kill -0 "$WATCH" 2>/dev/null; do sleep 120; done
echo "=== matched-K finished $(date '+%F %H:%M'); re-running K=4, K=6 and K=8 sub 1 ==="
RUNS="4:3 6:1 8:1" bash "$EXP/run_matchedK.sh"
echo "=== rebuilding report + figure $(date '+%F %H:%M') ==="
"$PY" "$EXP/verify_rerun.py"
"$PY" "$EXP/attenuation_report.py"
"$PY" "$EXP/make_figure5.py"
echo "=== done $(date '+%F %H:%M') ==="

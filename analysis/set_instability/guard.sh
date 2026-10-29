#!/bin/bash
#
# Detached safety guard for the E0 run. Launch with:
#   nohup bash guard.sh >/dev/null 2>&1 &
#
# Why this exists separately from the harness Monitor: the Monitor is a
# convenience notifier and has been observed to get recycled (~2h, exit 144).
# The disk guard is safety-critical -- if pruning silently stops working, each
# seed keeps 26GB instead of ~1GB and four seeds eat ~104GB that E1/E2 need.
# So the guard runs as a plain detached process and appends events to
# logs/guard.log; the Monitor just tails that file. If the Monitor dies, the
# guard keeps protecting the run.
#
# Checks:
#   post-prune  free < 230GB *at the moment a seed completes* -> pruning failed.
#               Only meaningful here: mid-seed the run legitimately holds 3
#               unpruned saves (~26GB), bottoming out near 216GB.
#   hard floor  free < 180GB at any time -> genuinely running out of space.

set -u

EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
export PATH="${VLLM_ENV_BIN:-$HOME/miniconda3/envs/vllm/bin}:$PATH"

BASELINE=242
PRUNE_FLOOR=230
HARD_FLOOR=180
LOG="$EXP/logs/guard.log"

say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" >>"$LOG"; }

free_gb() { df -BG --output=avail "$REPO" 2>/dev/null | tail -1 | tr -dc '0-9'; }
seeds_done() { echo $(( $(wc -l <"$EXP/manifest.tsv" 2>/dev/null || echo 1) - 1 )); }

stop_e0() {
    pkill -f 'run_all[.]sh'  2>/dev/null || true
    pkill -f 'run_seed[.]sh' 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
    pkill -f 'ray::' 2>/dev/null || true
    du -sh "$REPO"/checkpoints/*_seed* 2>/dev/null | while read -r l; do say "  $l"; done
}

prev_seeds=$(seeds_done)
peak_gpu=0
say "GUARD START | ${prev_seeds} seed(s) done | $(free_gb)GB free | prune floor ${PRUNE_FLOOR}GB, hard floor ${HARD_FLOOR}GB"

while true; do
    free=$(free_gb); [ -z "$free" ] && free=$BASELINE

    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    case "$g" in ''|*[!0-9]*) g=0 ;; esac
    [ "$g" -gt "$peak_gpu" ] && peak_gpu=$g

    if [ "$free" -lt "$HARD_FLOOR" ]; then
        say "CRITICAL: free ${free}GB < hard floor ${HARD_FLOOR}GB -- out of space. Stopping E0."
        stop_e0; say "GUARD EXIT (hard floor)"; exit 1
    fi

    n=$(seeds_done)
    if [ "$n" -gt "$prev_seeds" ]; then
        # manifest is written after pruning in run_seed.sh, so disk is meaningful now
        if [ "$free" -lt "$PRUNE_FLOOR" ]; then
            say "CRITICAL: seed completed but free is ${free}GB (< ${PRUNE_FLOOR}GB) -- PRUNING FAILED. Stopping E0 to protect space for E1/E2."
            stop_e0; say "GUARD EXIT (pruning failed)"; exit 1
        fi
        say "SEED DONE: manifest has ${n} seeds | free ${free}GB -- pruning OK | peak GPU ${peak_gpu}MiB of 24564"
        prev_seeds=$n
        peak_gpu=0
    fi

    if ! pgrep -f 'run_all[.]sh' >/dev/null 2>&1; then
        if grep -q 'all seeds done' "$EXP/logs/run_all.log" 2>/dev/null; then
            say "E0 COMPLETE: ${n} seeds | free ${free}GB. Next: bash classify.sh && python analyze_stability.py"
        else
            say "E0 ENDED UNEXPECTEDLY: ${n} seeds | free ${free}GB. Check logs/run_all.log"
        fi
        say "GUARD EXIT (run finished)"; exit 0
    fi

    sleep 60
done

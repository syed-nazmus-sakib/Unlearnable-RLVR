#!/bin/bash
#
# Runs the matched-K / attenuation work as soon as E2's seeds release the GPU.
#
#   SEEDS_PID=<pid> setsid nohup ./chain_matchedK.sh </dev/null >> logs_chain_matchedK.out 2>&1 &
#
# Waits on the process, not its status: if a seed fails, the confound panel still
# gets measured. A12.5 puts these runs after the seeds, and that ordering holds.
set -uo pipefail
EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WATCH="${SEEDS_PID:?set SEEDS_PID to the run_all_seeds.sh pid}"

echo "=== matched-K chain armed $(date '+%F %H:%M'), waiting on seeds pid $WATCH ==="
while kill -0 "$WATCH" 2>/dev/null; do sleep 120; done
echo "=== seeds released the GPU at $(date '+%F %H:%M') ==="

for _ in $(seq 1 30); do
    busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
    [[ -z "$busy" ]] && break
    echo "GPU still busy (${busy//$'\n'/ }) -- waiting"
    sleep 60
done

bash "$EXP/run_matchedK.sh"
echo "=== matched-K chain finished $(date '+%F %H:%M') rc=$? ==="

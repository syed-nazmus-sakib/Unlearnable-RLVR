#!/bin/bash
#
# Live E0 status dashboard.
#
#   bash watch_e0.sh          refresh every 15s until you Ctrl-C
#   bash watch_e0.sh --once   print one snapshot and exit
#
# Reads the same logs the Monitor watches; it is purely a viewer and never
# touches the run. Safe to start, stop, and restart at any time.

EXP="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$EXP/../../unlearnability-rlvr" && pwd)}"
TARGET_STEP="${TARGET_STEP:-120}"
ALL_SEEDS="2 3 4 5"

snapshot() {
    local now seed L last_step score ent st free done_n remain eta_min eta_seed
    now="$(date +%H:%M:%S)"

    # which seed is running (run_seed.sh's last argument)
    seed="$(pgrep -af 'run_seed[.]sh' 2>/dev/null | grep -oE '[0-9]+$' | head -1)"
    done_n=$(( $(wc -l <"$EXP/manifest.tsv" 2>/dev/null || echo 1) - 1 ))
    free="$(df -BG --output=avail "$REPO" 2>/dev/null | tail -1 | tr -dc '0-9')"

    printf '\033[1mE0 seed stability\033[0m   %s\n' "$now"
    printf '%s\n' "--------------------------------------------------------"

    if [ -z "$seed" ]; then
        if grep -q 'all seeds done' "$EXP/logs/run_all.log" 2>/dev/null; then
            printf '  status   : \033[32mCOMPLETE\033[0m\n'
        else
            printf '  status   : \033[31mnot running\033[0m (check logs/run_all.log)\n'
        fi
    else
        L="$EXP/logs/train_seed${seed}.log"
        last_step="$(grep -aoE "step:[0-9]+ - response_length" "$L" 2>/dev/null | tail -1 | grep -oE '[0-9]+')"
        score="$(grep -aoE 'critic/score/mean:[0-9.]+' "$L" 2>/dev/null | tail -1 | cut -d: -f2)"
        ent="$(grep -aoE 'actor/entropy_loss:[-0-9.]+' "$L" 2>/dev/null | tail -1 | cut -d: -f2)"
        st="$(grep -aoE 'timing_s/step:[0-9.]+' "$L" 2>/dev/null | tail -1 | cut -d: -f2)"
        [ -z "$last_step" ] && last_step=0
        [ -z "$st" ] && st=53

        remain=$(( TARGET_STEP - last_step ))
        [ "$remain" -lt 0 ] && remain=0
        eta_seed=$(awk -v r="$remain" -v s="$st" 'BEGIN{printf "%.0f", r*s/60 + 30}')  # +30min eval

        printf '  seed     : \033[1m%s\033[0m   (%s of 4 done)\n' "$seed" "$done_n"
        printf '  step     : %s / %s' "$last_step" "$TARGET_STEP"
        awk -v a="$last_step" -v b="$TARGET_STEP" 'BEGIN{
            n=int(40*a/b); printf "  ["
            for(i=0;i<40;i++) printf (i<n ? "#" : ".")
            printf "] %.0f%%\n", 100*a/b }'
        printf '  reward   : %s      entropy: %s      %ss/step\n' "${score:-?}" "${ent:-?}" "${st:-?}"
        printf '  this seed: ~%s min left (incl. eval)\n' "$eta_seed"

        # remaining whole seeds after this one
        local later=0 s2
        for s2 in $ALL_SEEDS; do [ "$s2" -gt "$seed" ] && later=$((later+1)); done
        eta_min=$(awk -v e="$eta_seed" -v l="$later" 'BEGIN{printf "%.1f", (e + l*162)/60}')
        printf '  all E0   : ~%s h left  (%s seed(s) still queued)\n' "$eta_min" "$later"
    fi

    printf '  disk     : %sGB free' "${free:-?}"
    if [ -n "$free" ] && [ "$free" -lt 230 ]; then
        printf '   (mid-seed dip, expected)\n'
    else
        printf '   (baseline 242GB)\n'
    fi
    printf '  gpu      : %s\n' "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)"
    printf '%s\n' "--------------------------------------------------------"
    grep -E 'ok in |FAILED|all seeds done' "$EXP/logs/run_all.log" 2>/dev/null | tail -4 | sed 's/^/  /'
}

if [ "${1:-}" = "--once" ]; then
    snapshot
else
    trap 'printf "\n"; exit 0' INT
    while true; do clear; snapshot; printf '\n  refreshing every 15s -- Ctrl-C to stop\n'; sleep 15; done
fi

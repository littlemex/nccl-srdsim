#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run a GPU test (tests/gpu/gin_signal_order.cu or tests/gpu/gin_get.cu) under the plugin on one GPU host, in each
# delivery mode, and check the outcome each mode must produce: no wrong element in a mode that keeps the test's
# guarantee, and at least one seed with a wrong element in a negative control. The controls are violate-i1, and any
# mode written control:MODE, which runs MODE with GIN_GET_CHECK_EARLY=1 and the variables in CONTROL_ENV.
#
# Each rank is a separate process on GPU 0 with its own NCCL_HOSTID, so NCCL treats every rank as a
# separate node rather than reach the others through peer memory. An MPS server runs
# the ranks' kernels concurrently instead of time-slicing them.
#
# Usage: tests/gpu/run.sh PLUGIN_SO TEST_BINARY [NRANKS] [SEEDS]
# Environment: RUN_TIMEOUT (seconds per run, default 180), WAIT_TIMEOUT (seconds a kernel waits for one
# signal before it gives up, default 60), ROUNDS (per seed, default 50), MODES (space-separated, default
# every mode), CONTROL_ENV (NAME=VALUE pairs for the control: runs).
set -uo pipefail
PLUGIN="${1:?plugin .so}"; TEST="${2:?test binary}"; NRANKS="${3:-3}"; SEEDS="${4:-5}"
RUN_TIMEOUT="${RUN_TIMEOUT:-180}"; WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"; ROUNDS="${ROUNDS:-50}"
MODES="${MODES:-srd inorder peer-seq reverse-ops signal-last defer-one violate-i1}"
CONTROL_ENV="${CONTROL_ENV:-}"
is_control() { [ "$1" = violate-i1 ] || [ "${1#control:}" != "$1" ]; }
for f in "$PLUGIN" "$TEST"; do [ -e "$f" ] || { echo "no such file: $f" >&2; exit 2; }; done
command -v nvidia-cuda-mps-control >/dev/null || { echo "nvidia-cuda-mps-control is not on PATH" >&2; exit 2; }
# Matched on the full command line: process names are cut at 15 characters, so -x on the name never matches.
if pgrep -f '(^|/)nvidia-cuda-mps-(control|server)( |$)' >/dev/null; then
    echo "an MPS daemon is already running on this host; stop it first (echo quit | nvidia-cuda-mps-control)" >&2
    exit 2
fi

WORK=$(mktemp -d)
export CUDA_MPS_PIPE_DIRECTORY="$WORK/mps-pipe" CUDA_MPS_LOG_DIRECTORY="$WORK/mps-log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
RANK_PIDS=()
RANK_STATUS=""
FIRST_STATUS=""
fail=0

kill_ranks() {
    local p
    for p in "${RANK_PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
    for p in "${RANK_PIDS[@]}"; do wait "$p" 2>/dev/null; done
    RANK_PIDS=()
}
mps_start() { nvidia-cuda-mps-control -d || { echo "could not start an MPS server" >&2; exit 2; }; }
mps_stop() { echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; }
cleanup() {
    local rc=$?
    kill_ranks
    mps_stop
    if [ "$rc" -eq 0 ] && [ "$fail" -eq 0 ]; then rm -rf "$WORK"; else echo "logs kept in $WORK"; fi
}
trap cleanup EXIT
trap 'fail=1; exit 130' INT TERM
mps_start

# Runs every rank of one (mode, seed). Leaves one exit status per rank in RANK_STATUS and, in
# FIRST_STATUS, the status of the first rank that exited with something other than 0 or 1 (else the
# worst of those). It runs in this shell, not in a command substitution, so that the trap can see and
# stop the ranks. A rank that crashed makes the others wait for it until their own timeout, so they are
# stopped at once; a rank that timed out (3) gives the others GRACE seconds to time out and log too.
GRACE=10
run_once() {
    local mode=$1 seed=$2 id="$WORK/id-$1-$2" r p q alive rc stop_at=-1
    local -a status extra=()
    # control:MODE runs MODE with the test's own negative control switched on.
    if [ "${mode#control:}" != "$mode" ]; then
        extra=(GIN_GET_CHECK_EARLY=1)
        # shellcheck disable=SC2206
        extra+=($CONTROL_ENV)
    fi
    RANK_PIDS=(); FIRST_STATUS=""
    for r in $(seq 0 $((NRANKS - 1))); do
        env "${extra[@]}" NCCL_RMA_PLUGIN="$PLUGIN" NCCL_GIN_TYPE=2 NCCL_HOSTID="srdsim-rank-$r" \
        SRDSIM_MODE="${mode#control:}" SRDSIM_SEED="$seed" \
            timeout -k 10 "$RUN_TIMEOUT" "$TEST" "$r" "$NRANKS" "$id" "$ROUNDS" 65536 16 "$WAIT_TIMEOUT" \
            > "$WORK/log-$mode-$seed-$r" 2>&1 &
        RANK_PIDS+=($!)
        status[$r]=""
    done
    while :; do
        alive=0
        for r in "${!RANK_PIDS[@]}"; do
            [ -n "${status[$r]}" ] && continue
            p=${RANK_PIDS[$r]}
            if kill -0 "$p" 2>/dev/null; then alive=1; continue; fi
            wait "$p"; rc=$?; status[$r]=$rc
            if [ "$rc" -gt 1 ] && [ -z "$FIRST_STATUS" ]; then
                FIRST_STATUS=$rc
                if [ "$rc" -eq 3 ]; then stop_at=$((SECONDS + GRACE)); else stop_at=$SECONDS; fi
            fi
        done
        [ "$alive" -eq 0 ] && break
        if [ "$stop_at" -ge 0 ] && [ "$SECONDS" -ge "$stop_at" ]; then
            for q in "${RANK_PIDS[@]}"; do kill -TERM "$q" 2>/dev/null; done
            stop_at=-1
        fi
        sleep 0.2
    done
    RANK_PIDS=()
    RANK_STATUS="${status[*]}"
    if [ -z "$FIRST_STATUS" ]; then
        FIRST_STATUS=0; for rc in $RANK_STATUS; do [ "$rc" -gt "$FIRST_STATUS" ] && FIRST_STATUS=$rc; done
    fi
}

for mode in $MODES; do
    per_seed=(); bad_seeds=(); found=0
    for seed in $(seq 1 "$SEEDS"); do
        run_once "$mode" "$seed"
        worst=$FIRST_STATUS
        per_seed+=("$worst")
        # A rank that was stopped mid-kernel may leave the MPS server holding its context; start a fresh one.
        restart=0; [ "$worst" -ge 3 ] && restart=1
        for rc in $RANK_STATUS; do [ "$rc" -ge 124 ] && restart=1; done
        if [ "$restart" -eq 1 ]; then mps_stop; mps_start; fi
        if is_control "$mode"; then
            [ "$worst" -le 1 ] || bad_seeds+=("$seed")
            [ "$worst" -eq 1 ] && found=1
        else
            [ "$worst" -eq 0 ] || bad_seeds+=("$seed")
        fi
    done
    if is_control "$mode"; then
        expect="some seed finds a wrong element"
        if [ "${#bad_seeds[@]}" -eq 0 ] && [ "$found" -eq 1 ]; then verdict=PASS; else verdict=FAIL; fi
    else
        expect="no wrong element"
        if [ "${#bad_seeds[@]}" -eq 0 ]; then verdict=PASS; else verdict=FAIL; fi
    fi
    printf '%-12s exit per seed: %-20s expected: %-34s %s\n' "$mode" "${per_seed[*]}" "$expect" "$verdict"
    for seed in $(seq 1 "$SEEDS"); do
        grep -h "wrong elements" "$WORK/log-$mode-$seed-0" 2>/dev/null | head -1 | sed "s/^/    seed $seed: /"
    done
    if [ "$verdict" = FAIL ]; then
        fail=1
        for seed in "${bad_seeds[@]}"; do
            for f in "$WORK"/log-"$mode"-"$seed"-*; do echo "    == $f"; tail -5 "$f" | sed 's/^/    /'; done
        done
    fi
done
exit "$fail"

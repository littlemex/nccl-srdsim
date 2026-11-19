#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run tests/gpu/gin_signal_order.cu under the plugin on one GPU host, in every delivery mode, and check
# the outcome each mode must produce: no wrong element in the modes that keep I1, and at least one seed
# with a wrong element under violate-i1.
#
# Each rank is a separate process on GPU 0 with its own NCCL_HOSTID, so NCCL treats every rank as a
# separate node and reaches every peer through GIN rather than through peer memory. An MPS server lets
# the ranks' kernels run at the same time, which they need because each waits for the others.
#
# Usage: tests/gpu/run.sh PLUGIN_SO TEST_BINARY [NRANKS] [SEEDS]
set -uo pipefail
PLUGIN="${1:?plugin .so}"; TEST="${2:?test binary}"; NRANKS="${3:-3}"; SEEDS="${4:-5}"
MODES_KEEP="srd inorder peer-seq reverse-ops signal-last defer-one"
WORK=$(mktemp -d)
trap 'echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; rm -rf "$WORK"' EXIT

export CUDA_MPS_PIPE_DIRECTORY="$WORK/mps-pipe" CUDA_MPS_LOG_DIRECTORY="$WORK/mps-log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d || { echo "could not start an MPS server"; exit 2; }

run_once() {   # mode seed -> prints the worst exit status of the ranks
    local mode=$1 seed=$2 id="$WORK/id-$1-$2" worst=0 pids=() r rc
    for r in $(seq 0 $((NRANKS - 1))); do
        NCCL_RMA_PLUGIN="$PLUGIN" NCCL_GIN_TYPE=2 NCCL_HOSTID="srdsim-rank-$r" \
        SRDSIM_MODE="$mode" SRDSIM_SEED="$seed" \
            timeout 600 "$TEST" "$r" "$NRANKS" "$id" > "$WORK/log-$mode-$seed-$r" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; rc=$?; [ "$rc" -gt "$worst" ] && worst=$rc; done
    echo "$worst"
}

fail=0
for mode in $MODES_KEEP violate-i1; do
    results=()
    for seed in $(seq 1 "$SEEDS"); do results+=("$(run_once "$mode" "$seed")"); done
    summary="${results[*]}"
    if [ "$mode" = violate-i1 ]; then
        if [[ " $summary " == *" 1 "* ]] && [[ " $summary " != *" 2 "* ]] && [[ " $summary " != *" 124 "* ]]; then verdict=PASS; else verdict=FAIL; fail=1; fi
        expect="some seed finds a wrong element"
    else
        if [[ "$summary" =~ ^(0 ?)+$ ]]; then verdict=PASS; else verdict=FAIL; fail=1; fi
        expect="no wrong element"
    fi
    printf '%-12s exit per seed: %-20s expected: %-34s %s\n' "$mode" "$summary" "$expect" "$verdict"
    for f in "$WORK"/log-"$mode"-*-0; do grep -h "wrong elements" "$f" | head -1; done | sed 's/^/    /'
done
if [ "$fail" -ne 0 ]; then
    echo "logs of the last failing runs:"; for f in "$WORK"/log-*; do echo "== $f"; tail -5 "$f"; done
fi
exit "$fail"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Runs libfabric's fabtests between two processes on one host with no EFA device, through an unmodified
# efa provider on the device stand-in in src/device, with every transfer's data checked (-v).
#
#   ci/efa_fabtests.sh LIBFABRIC_SRC WORK
#
# LIBFABRIC_SRC is a clone of libfabric at the pinned commit, not configured yet: a build configured for
# the efa unit tests replaces the provider's data path with stubs, so this one is configured without
# them. WORK receives the stand-in, the installed libfabric and fabtests, and the logs.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 LIBFABRIC_SRC WORK" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC=$(cd "$1" && pwd)
mkdir -p "$2"; WORK=$(cd "$2" && pwd)

cd "$SRC"
[ -x configure ] || ./autogen.sh > "$WORK/autogen.log" 2>&1
./configure --quiet --prefix="$WORK/libfabric" --enable-efa=yes \
    --disable-psm3 --disable-opx --disable-verbs --disable-usnic --disable-lpp > "$WORK/configure.log" 2>&1 \
    || { tail -30 "$WORK/configure.log"; exit 1; }
make -j"$(nproc)" install > "$WORK/make.log" 2>&1 || { grep -B2 -A6 -iE 'error' "$WORK/make.log" | head -40; exit 1; }
cd fabtests
[ -x configure ] || ./autogen.sh > "$WORK/fabtests-autogen.log" 2>&1
./configure --quiet --with-libfabric="$WORK/libfabric" --prefix="$WORK/fabtests" > "$WORK/fabtests-configure.log" 2>&1 \
    || { tail -30 "$WORK/fabtests-configure.log"; exit 1; }
make -j"$(nproc)" install > "$WORK/fabtests-make.log" 2>&1 || { grep -B2 -A6 -iE 'error' "$WORK/fabtests-make.log" | head -40; exit 1; }

make -C "$ROOT/src/device" OUT="$WORK/device" > "$WORK/device-build.log" 2>&1 || { cat "$WORK/device-build.log"; exit 1; }

export LD_LIBRARY_PATH="$WORK/device:$WORK/libfabric/lib"
# Two processes on one host share a GID; without this the provider would send between them through shm.
export FI_EFA_ENABLE_SHM_TRANSFER=0
FT=$WORK/fabtests/bin
fail=0
n=0
run() {
    local name=$1 log
    shift
    n=$((n + 1))
    log="$WORK/fabtest-$n"
    timeout 300 "$FT/$name" "$@" -E > "$log.server" 2>&1 & local server=$!
    sleep 1
    local c=0 s=0
    timeout 300 "$FT/$name" "$@" -E 127.0.0.1 > "$log.client" 2>&1 || c=$?
    wait "$server" || s=$?
    if [ "$c" -eq 0 ] && [ "$s" -eq 0 ]; then
        echo "PASS $name $*"
    else
        echo "FAIL $name $* (server $s, client $c)"
        tail -5 "$log.client" "$log.server"
        fail=1
    fi
}
run fi_rdm_pingpong -p efa -f efa-direct -I 50 -S all -v
run fi_rdm_pingpong -p efa -f efa -I 50 -S all -v
run fi_rdm_tagged_pingpong -p efa -f efa -I 50 -S all -v
run fi_rdm_tagged_bw -p efa -f efa -I 50 -S all -v
run fi_rma_bw -p efa -f efa -e rdm -o write -I 50 -S all -v
run fi_rma_bw -p efa -f efa -e rdm -o read -I 50 -S all -v
run fi_rma_bw -p efa -f efa -e rdm -o writedata -I 50 -S all -v
run fi_rma_bw -p efa -f efa-direct -e rdm -o write -I 50 -S all -v
run fi_rma_bw -p efa -f efa-direct -e rdm -o read -I 50 -S all -v
exit "$fail"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Runs libfabric's fabtests between two processes on one host with no EFA device, through an unmodified
# efa provider on the device stand-in in src/device, with every transfer's data checked (-v). Then checks
# that messages arrive in the order sent (tests/device/msg_order.c, EFA-MSG-001) while the stand-in
# reorders them as SRD may, and, as the mutation control, that a provider built not to restore the order
# fails that check on the stand-in and passes it when the stand-in keeps the order: a fake that never
# reorders would hide this bug.
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
"$ROOT/ci/device_symbols.sh" "$WORK/libfabric/lib/libfabric.so" "$WORK/device"

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

cc -O2 -Wall -Wextra -Werror -I"$WORK/libfabric/include" "$ROOT/tests/device/msg_order.c" \
    -L"$WORK/libfabric/lib" -lfabric -o "$WORK/msg_order"
# order MODE LIBRARY_DIR: the exit status of msg_order with the stand-in in MODE below that libfabric
order() {
    local st=0
    rm -f "$WORK/stats-$1"
    SRDSIM_MODE=$1 SRDSIM_DEVICE_STATS="$WORK/stats-$1" LD_LIBRARY_PATH="$WORK/device:$2" \
        timeout 300 "$WORK/msg_order" 2000 > "$WORK/order-$1.log" 2>&1 || st=$?
    tail -2 "$WORK/order-$1.log"
    return "$st"
}
if order srd "$WORK/libfabric/lib"; then
    reordered=$(awk '{r += $10} END {print r + 0}' "$WORK/stats-srd" 2>/dev/null || echo 0)
    if [ "$reordered" -gt 0 ]; then
        echo "PASS messages arrive in order while the stand-in reorders them ($reordered frames reordered)"
    else
        echo "FAIL the stand-in reordered nothing, so the order check proved nothing"; fail=1
    fi
else
    echo "FAIL messages out of order"; fail=1
fi

# The mutation: the provider takes every message as the one it expects next, so it never holds one back.
cd "$SRC"
sed -i 's/if (ofi_recvwin_is_exp(robuf, msg_id))/if (1 || ofi_recvwin_is_exp(robuf, msg_id))/' prov/efa/src/rdm/efa_rdm_peer.c
grep -q '1 || ofi_recvwin_is_exp' prov/efa/src/rdm/efa_rdm_peer.c || { echo "the mutation did not apply"; exit 1; }
make -j"$(nproc)" > "$WORK/make-mutated.log" 2>&1 || { tail -20 "$WORK/make-mutated.log"; exit 1; }
git checkout -q prov/efa/src/rdm/efa_rdm_peer.c
if order srd "$SRC/src/.libs"; then
    echo "FAIL the order check passed against a provider that does not restore the order"; fail=1
else
    echo "PASS the order check catches a provider that does not restore the order"
fi
if order inorder "$SRC/src/.libs"; then
    echo "PASS that provider passes when the stand-in keeps the order, so only reordering exposes it"
else
    echo "FAIL that provider failed even in order; the control does not isolate reordering"; fail=1
fi
# This project's own fi tests, against the efa provider on the stand-in instead of this project's provider.
# Not run here, each for a reason that is about the test, not the stand-in: directsend, closewrite, inorder128,
# test_fi_provider and test_fi_wire check behaviour of this project's provider itself; unreachable expects the
# status a device gives for a QP nobody listens on, and closeinject expects an inject posted just before its
# endpoint closes to arrive, neither yet seen on hardware (#81); test_fi_cq_threads completion checks that both of
# its reading threads got a completion, which depends on scheduling, not on the provider; test_fi_close_while_reading
# closes endpoints while another thread reads their CQ, which FI_THREAD_COMPLETION leaves to the application to
# serialize (EFA-THR-002).
export PKG_CONFIG_PATH="$WORK/libfabric/lib/pkgconfig"
cmake -S "$ROOT" -B "$WORK/fi-build" -DSRDSIM_BUILD_TESTS=OFF -DSRDSIM_BUILD_FI_TEST=ON -DCMAKE_CXX_FLAGS=-Werror \
    -DSRDSIM_FI_PROVIDER=/nonexistent/none.so > "$WORK/fi-cmake.log" 2>&1 \
    || { tail -20 "$WORK/fi-cmake.log"; exit 1; }
cmake --build "$WORK/fi-build" -j \
    --target test_fi_transfer test_fi_multipeer test_fi_cq_threads > "$WORK/fi-build.log" 2>&1 \
    || { grep -m5 -B2 -A5 error "$WORK/fi-build.log"; exit 1; }
fi_test() {
    local st=0 log="$WORK/fi-$n.log"
    n=$((n + 1))
    timeout 120 "$WORK/fi-build/$1" /nonexistent/none.so "${@:2}" > "$log" 2>&1 || st=$?
    if [ "$st" -eq 0 ]; then
        echo "PASS $*"
    else
        echo "FAIL $* (exit $st)"; grep -E 'FAIL' "$log" | head -3; fail=1
    fi
}
for s in sendrecv tagged inject truncate read badkey writedata reorder; do
    fi_test test_fi_transfer "$s"
done
fi_test test_fi_multipeer
fi_test test_fi_cq_threads domain
exit "$fail"

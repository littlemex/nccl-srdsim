#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Runs libfabric's own efa unit tests on a host with no EFA device, with the device stand-in in src/device
# below an unmodified efa provider.
#
#   ci/efa_unit_tests.sh LIBFABRIC_SRC WORK
#
# LIBFABRIC_SRC is a clone of libfabric at the pinned commit; it is configured and built in place. WORK
# receives the stand-in, the test results and the logs. Needs libibverbs-dev and libcmocka-dev.
#
# Three runs: without the stand-in, which must find no device (the negative control); with it, where
# every failure must be one that tests/efa_unit/expected lists with its reason; and with a stand-in that
# lets a CQ be destroyed while a QP still uses it, which the real verbs refuse with EBUSY, where the
# upstream test of that rule must fail (the mutation control).
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 LIBFABRIC_SRC WORK" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC=$(cd "$1" && pwd)
mkdir -p "$2"; WORK=$(cd "$2" && pwd)

# configure takes one directory with include/ and lib/ below it; a distribution splits them.
CMOCKA=$WORK/cmocka
mkdir -p "$CMOCKA/include" "$CMOCKA/lib"
cp /usr/include/cmocka*.h "$CMOCKA/include/"
cp -a "$(dirname "$(ldconfig -p | awk '/libcmocka\.so\.0 / {print $NF; exit}')")"/libcmocka.so* "$CMOCKA/lib/"

cd "$SRC"
[ -x configure ] || ./autogen.sh > "$WORK/autogen.log" 2>&1
./configure --quiet --enable-efa=yes --enable-efa-unit-test="$CMOCKA" \
    --disable-psm3 --disable-opx --disable-verbs --disable-usnic --disable-lpp > "$WORK/configure.log" 2>&1 \
    || { tail -30 "$WORK/configure.log"; exit 1; }
grep -E '^#define HAVE_EFA_DATA_PATH_DIRECT ' config.h
make -j"$(nproc)" > "$WORK/make.log" 2>&1 || { grep -B2 -A6 -iE 'error' "$WORK/make.log" | head -40; exit 1; }
TEST=$SRC/prov/efa/test/efa_unit_test
[ -x "$TEST" ] || { echo "the efa unit tests were not built"; exit 1; }

make -C "$ROOT/src/device" OUT="$WORK/device" > "$WORK/device-build.log" 2>&1 || { cat "$WORK/device-build.log"; exit 1; }
make -s -C "$ROOT/src/device" features

# run OUT LIBRARY_PATH: cmocka writes JUnit XML to stdout; a crash still leaves the cases that finished.
run() {
    LD_LIBRARY_PATH=$2 CMOCKA_MESSAGE_OUTPUT=xml timeout 1500 "$TEST" > "$1" 2> "$1.stderr" || true
}

run "$WORK/without-device.xml" "$CMOCKA/lib"
python3 "$ROOT/tools/efa_unit_tests.py" no-device "$WORK/without-device.xml"

run "$WORK/with-device.xml" "$WORK/device:$CMOCKA/lib"
LD_LIBRARY_PATH="$WORK/device:$CMOCKA/lib" ldd "$TEST" | grep -E 'libibverbs|libefa'
python3 "$ROOT/tools/efa_unit_tests.py" check "$WORK/with-device.xml" "$ROOT/tests/efa_unit/expected"

make -C "$ROOT/src/device" OUT="$WORK/device-mutated" MUTATION=-DSD_MUTATION_CQ_IN_USE > "$WORK/device-mutated-build.log" 2>&1 \
    || { cat "$WORK/device-mutated-build.log"; exit 1; }
run "$WORK/mutated.xml" "$WORK/device-mutated:$CMOCKA/lib"
python3 "$ROOT/tools/efa_unit_tests.py" fails "$WORK/mutated.xml" test_efa_cq_close_returns_ebusy_with_bound_ep \
    || { echo "the upstream tests did not notice a CQ destroyed while in use"; exit 1; }

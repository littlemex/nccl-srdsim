#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds libsrdsim-fi.so, the libfabric provider in src/fi, inside a libfabric source tree, the way
# libfabric builds its own providers as DSOs, and installs that libfabric under PREFIX.
#
#   scripts/build_fi_provider.sh LIBFABRIC_SRC PREFIX [CONFIGURE_ARGS...]
#
# LIBFABRIC_SRC is a clone of libfabric at the commit to build against; it is modified in place (the
# provider is added to it). The provider ends up in PREFIX/lib/srdsim/libsrdsim-fi.so, outside libfabric's
# default provider directory, so a process loads it only when FI_PROVIDER_PATH names it.
#
# The build happens inside LIBFABRIC_SRC, so building a second configuration in the same tree would
# otherwise link the first one's object files -- which is silent, and the usual outcome is a provider
# with no sanitizer instrumentation whose tests all pass for the wrong reason. This script records the
# configuration it built with and cleans the tree when the next one differs.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 LIBFABRIC_SRC PREFIX [CONFIGURE_ARGS...]" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC=$(cd "$1" && pwd); PREFIX=$2; shift 2
[ -f "$SRC/configure.ac" ] && [ -f "$SRC/Makefile.am" ] || { echo "$SRC is not a libfabric source tree" >&2; exit 2; }

rm -rf "$SRC/prov/srdsim"
mkdir -p "$SRC/prov/srdsim"
cp "$ROOT"/src/fi/* "$SRC/prov/srdsim/"
if ! grep -q 'FI_PROVIDER_SETUP(\[srdsim\])' "$SRC/configure.ac"; then
    grep -q '^FI_PROVIDER_FINI$' "$SRC/configure.ac" || { echo "configure.ac has no FI_PROVIDER_FINI line" >&2; exit 1; }
    sed -i 's/^FI_PROVIDER_FINI$/FI_PROVIDER_SETUP([srdsim])\nFI_PROVIDER_FINI/' "$SRC/configure.ac"
fi
if ! grep -q '^include prov/srdsim/Makefile.include$' "$SRC/Makefile.am"; then
    grep -q '^include prov/udp/Makefile.include$' "$SRC/Makefile.am" || { echo "Makefile.am has no udp include to follow" >&2; exit 1; }
    sed -i 's|^include prov/udp/Makefile.include$|&\ninclude prov/srdsim/Makefile.include|' "$SRC/Makefile.am"
fi

cd "$SRC"
# What this invocation asks for that affects the object files: the configure arguments and the
# compiler and flags, which arrive as trailing VAR=value arguments. PREFIX is left out on purpose --
# it decides where the result is installed, not how it is compiled, so building the same
# configuration into a second prefix must not throw the objects away.
STAMP=.srdsim-build-configuration
WANT="--enable-srdsim=dl $*"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$WANT" ]; then
    echo "this tree was built with a different configuration; cleaning it first" >&2
    echo "  was:  $(cat "$STAMP")" >&2
    echo "  want: $WANT" >&2
    make distclean > distclean.log 2>&1 || rm -f "$STAMP"
fi
rm -f "$STAMP"
./autogen.sh > autogen.log 2>&1 || { tail -20 autogen.log; exit 1; }
./configure --quiet --prefix="$PREFIX" --enable-srdsim=dl "$@" > configure.log 2>&1 || { tail -30 configure.log; exit 1; }
grep -q 'DSO providers:.*srdsim' configure.log || { echo "configure did not select the srdsim provider:"; grep -A3 'providers:' configure.log; exit 1; }
make -j"$(nproc)" > make.log 2>&1 || { grep -B2 -A8 'error' make.log | head -60; exit 1; }
# Warnings in the provider's own files fail the build; libfabric's files are not this repository's to fix.
if grep -E 'prov/srdsim/[^:]+:[0-9]+:[0-9]+: warning' make.log; then echo "the provider builds with warnings" >&2; exit 1; fi
make install > install.log 2>&1 || { tail -20 install.log; exit 1; }
mkdir -p "$PREFIX/lib/srdsim"
mv "$PREFIX/lib/libfabric/libsrdsim-fi.so" "$PREFIX/lib/srdsim/libsrdsim-fi.so"
rm -f "$PREFIX/lib/libfabric/libsrdsim-fi.la"
printf '%s' "$WANT" > "$STAMP"
echo "$PREFIX/lib/srdsim/libsrdsim-fi.so"

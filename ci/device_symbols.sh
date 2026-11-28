#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails unless the device stand-in defines every libibverbs and libefa symbol a libfabric links against,
# with the same symbol version. A missing one is not seen at load time, because the dynamic linker binds
# functions when first called, and fails only when the provider takes the path that calls it.
#
#   ci/device_symbols.sh LIBFABRIC_SO DEVICE_DIR
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 LIBFABRIC_SO DEVICE_DIR" >&2; exit 2; }
want=$(nm -D --undefined-only "$1" | awk '{print $NF}' | grep -E '^(ibv_|efadv_)' | sort -u)
have=$(for l in "$2/libibverbs.so.1" "$2/libefa.so.1"; do
    objdump -T "$l" | awk '$4 != "*UND*" && $NF ~ /^(ibv_|efadv_)/ {print $NF "@" $(NF-1)}'
done | sort -u)
missing=$(comm -23 <(echo "$want") <(echo "$have"))
if [ -n "$missing" ]; then
    echo "the stand-in does not define what $1 links against:"
    echo "$missing"
    exit 1
fi
echo "the stand-in defines all $(echo "$want" | wc -l) libibverbs and libefa symbols $1 links against"

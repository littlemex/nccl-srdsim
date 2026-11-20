#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ci/check_exports.sh PLUGIN_SO NCCL_SRC
# Fails unless the plugin exports exactly the op tables ci/expected_exports.sh names for this release:
# every expected table present, every table listed as not exported absent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
so="${1:?usage: $0 PLUGIN_SO NCCL_SRC}"; src="${2:?usage: $0 PLUGIN_SO NCCL_SRC}"
defined=$(nm -D --defined-only "$so")
fail=0
while read -r s; do
    if [[ "$s" == !* ]]; then
        s=${s#!}
        if grep -Eq " [BDR] ${s}\$" <<<"$defined"; then
            echo "exported but listed in RMA_TABLES_NOT_EXPORTED: $s (remove it from versions.env)"; fail=1
        else
            echo "not exported, as versions.env records: $s"
        fi
    elif grep -Eq " [BDR] ${s}\$" <<<"$defined"; then
        echo "exported: $s"
    else
        echo "missing export: $s"; fail=1
    fi
done < <("$ROOT/ci/expected_exports.sh" "$src")
exit "$fail"

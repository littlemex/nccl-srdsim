#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ci/check_exports.sh PLUGIN_SO NCCL_SRC
# Fails unless the plugin's ncclRmaPlugin_vN exports are exactly those ci/expected_exports.sh names for
# this release: every expected table present, every table listed as not exported absent, and no other.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
so="${1:?usage: $0 PLUGIN_SO NCCL_SRC}"; src="${2:?usage: $0 PLUGIN_SO NCCL_SRC}"
expected=$("$ROOT/ci/expected_exports.sh" "$src")
exported=$(nm -D --defined-only "$so" | awk '$2 ~ /^[BDR]$/ && $3 ~ /^ncclRmaPlugin_v[0-9]+$/ {print $3}' | sort -u)
fail=0
while read -r s; do
    if [[ "$s" == !* ]]; then
        s=${s#!}
        if grep -qx "$s" <<<"$exported"; then
            echo "exported but listed in RMA_TABLES_NOT_EXPORTED: $s (remove it from versions.env)"; fail=1
        else
            echo "not exported, as versions.env records: $s"
        fi
    elif grep -qx "$s" <<<"$exported"; then
        echo "exported: $s"
    else
        echo "missing export: $s"; fail=1
    fi
done <<<"$expected"
for s in $exported; do
    grep -qx -e "$s" -e "!$s" <<<"$expected" || { echo "exported but not a table this release defines: $s"; fail=1; }
done
exit "$fail"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ci/check_serves_all.sh PLUGIN_SO
# The one build users run serves every pinned release: fails unless PLUGIN_SO, built against the newest
# release's headers, exports every table each release in NCCL_RELEASES requires (ci/expected_exports.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/versions.env"
so="${1:?usage: $0 PLUGIN_SO}"
exported=$(nm -D --defined-only "$so" | awk '$2 ~ /^[BDR]$/ {print $3}')
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
fail=0
for t in $NCCL_RELEASES; do
    v="NCCL_COMMIT_${t//[^A-Za-z0-9]/_}"
    git clone --quiet --depth 1 --branch "$t" https://github.com/NVIDIA/nccl.git "$work/$t"
    test "$(git -C "$work/$t" rev-parse HEAD)" = "${!v}" || { echo "$t is not the pinned commit ${!v}"; exit 1; }
    expected=$("$ROOT/ci/expected_exports.sh" "$work/$t")
    for s in $(grep -v '^!' <<<"$expected"); do
        if grep -qx "$s" <<<"$exported"; then echo "$t: finds $s"; else echo "$t: missing $s"; fail=1; fi
    done
done
exit "$fail"

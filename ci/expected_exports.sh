#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ci/expected_exports.sh NCCL_SRC
# Prints the op-table symbols the plugin must export for this NCCL release: ncclRmaPlugin_vN for every
# RMA table version N >= 14 whose header the release ships (src/include/plugin/rma/rma_vN.h), except
# those listed in RMA_TABLES_NOT_EXPORTED in versions.env. Prints those listed ones, prefixed with "!",
# so the caller can check they are still absent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/versions.env"
src="${1:?usage: $0 NCCL_SRC}"
found=0
for h in "$src"/src/include/plugin/rma/rma_v*.h; do
    n=$(basename "$h" .h); n=${n#rma_v}
    [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge 14 ] || continue
    found=1
    case " $RMA_TABLES_NOT_EXPORTED " in
        *" $n "*) echo "!ncclRmaPlugin_v$n" ;;
        *) echo "ncclRmaPlugin_v$n" ;;
    esac
done | sort -t v -k2 -n
[ -n "$(ls "$src"/src/include/plugin/rma/rma_v*.h 2>/dev/null)" ] || { echo "no RMA table headers under $src" >&2; exit 1; }

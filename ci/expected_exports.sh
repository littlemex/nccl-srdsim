#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ci/expected_exports.sh NCCL_SRC
# Prints the op-table symbols the plugin must export for this NCCL release: ncclRmaPlugin_vN for every
# RMA table version N >= RMA_TABLE_MIN whose header the release ships (src/include/plugin/rma/rma_vN.h),
# except those in RMA_TABLES_NOT_EXPORTED, which it prints prefixed with "!" so the caller can check
# they are absent. The headers stand in for NCCL's lookup list (src/plugin/rma.cc), which tries each
# version it defines, newest first.
set -euo pipefail
shopt -s nullglob
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/versions.env"
src="${1:?usage: $0 NCCL_SRC}"
out=()
for h in "$src"/src/include/plugin/rma/rma_v*.h; do
    n=${h##*/rma_v}; n=${n%.h}
    [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge "$RMA_TABLE_MIN" ] || continue
    case " $RMA_TABLES_NOT_EXPORTED " in
        *" $n "*) out+=("!ncclRmaPlugin_v$n") ;;
        *) out+=("ncclRmaPlugin_v$n") ;;
    esac
done
(( ${#out[@]} )) || { echo "no rma_vN.h with N >= $RMA_TABLE_MIN under $src/src/include/plugin/rma" >&2; exit 1; }
printf '%s\n' "${out[@]}" | sort -t v -k2 -n

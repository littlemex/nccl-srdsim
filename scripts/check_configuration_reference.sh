#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails when the source reads an SRDSIM_* variable that docs/configuration.md does not list, or when the
# document lists one the source no longer reads. Names in backticks in the document's tables count as
# listed; names in string literals under src/ count as read.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
read_vars=$(grep -ohE '"SRDSIM_[A-Z0-9_]+"' "$ROOT"/src/* | tr -d '"' | sort -u)
listed=$(grep -E '^\|' "$ROOT/docs/configuration.md" | grep -oE '`SRDSIM_[A-Z0-9_]+`' | tr -d '`' | sort -u)
missing=$(comm -23 <(echo "$read_vars") <(echo "$listed"))
stale=$(comm -13 <(echo "$read_vars") <(echo "$listed"))
[ -z "$missing" ] || { echo "read by src/ but not in docs/configuration.md:"; echo "$missing"; }
[ -z "$stale" ] || { echo "in docs/configuration.md but not read by src/:"; echo "$stale"; }
[ -z "$missing$stale" ] && echo "docs/configuration.md lists all $(echo "$read_vars" | wc -l | tr -d ' ') variables src/ reads"
[ -z "$missing$stale" ]

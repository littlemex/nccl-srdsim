#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails when the SRDSIM_* variables the source reads and the variables docs/configuration.md lists
# disagree, in either direction. Compares names only.
#
# A variable counts as read when its name is the literal first argument of one of the read helpers
# below, in a .cc or .h file under src/. A read whose name is not such a literal, or a name assembled
# from an "SRDSIM_" prefix, fails the check instead of being guessed at. A variable counts as listed
# when it is the first cell of a row of one of the document's tables.
set -euo pipefail
export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOC="$ROOT/docs/configuration.md"
READERS='getenv|env_str|env_size|impair_env'
LITERAL='"SRDSIM_[A-Z0-9_]+"'

fail=0
# Every call of a read helper outside comments, as file:line:text.
calls=$(grep -rnE --include='*.cc' --include='*.h' "\\b($READERS)[[:space:]]*\\(" "$ROOT/src" \
        | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true)
read_vars=$(grep -oE "\\b($READERS)[[:space:]]*\\([[:space:]]*$LITERAL" <<<"$calls" \
            | grep -oE 'SRDSIM_[A-Z0-9_]+' | sort -u || true)
# Calls that do not name a literal variable, other than the helpers' own definitions and the one
# getenv(name) each of them makes.
opaque=$(grep -vE "\\b($READERS)[[:space:]]*\\([[:space:]]*$LITERAL[[:space:]]*[,)]" <<<"$calls" \
         | grep -vE "\\b($READERS)\\(const char\\* name" \
         | grep -vE 'std::getenv\(name\)' || true)
built=$(grep -rnE --include='*.cc' --include='*.h' '"SRDSIM_"|"SRDSIM_[A-Z0-9_]*%' "$ROOT/src" || true)
listed=$(awk -F'|' '/^[[:space:]]*\|/ {print $2}' "$DOC" | grep -oE '`SRDSIM_[A-Z0-9_]+`' | tr -d '`' \
         | sort -u || true)

if [ -z "$read_vars" ] || [ -z "$listed" ]; then
    echo "found no variables read or none listed; the patterns in this script no longer match" >&2
    exit 1
fi
if [ -n "$opaque" ]; then
    echo "reads whose variable is not a literal name, which this check cannot follow:"; echo "$opaque"; fail=1
fi
if [ -n "$built" ]; then
    echo "variable names assembled at run time, which this check cannot follow:"; echo "$built"; fail=1
fi
missing=$(comm -23 <(echo "$read_vars") <(echo "$listed"))
stale=$(comm -13 <(echo "$read_vars") <(echo "$listed"))
if [ -n "$missing" ]; then echo "read by src/ but not a row of docs/configuration.md:"; echo "$missing"; fail=1; fi
if [ -n "$stale" ]; then echo "a row of docs/configuration.md but not read by src/:"; echo "$stale"; fail=1; fi
if [ "$fail" -eq 0 ]; then
    echo "docs/configuration.md has a row for each of the $(wc -l <<<"$read_vars" | tr -d ' ') variables src/ reads"
fi
exit "$fail"

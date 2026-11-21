#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails when the SRDSIM_* variables the source reads and the variables docs/configuration.md lists
# (and its Japanese translation, docs/configuration.ja.md) disagree, in either direction. Compares names only.
#
# A variable counts as read when its name is the literal first argument of one of the read helpers
# below, in a .cc or .h file under src/ (the only place the plugin and its headers live). A read whose
# name is not such a literal, a name assembled from an "SRDSIM_" prefix, and a quoted full name that no
# known helper takes all fail the check instead of being guessed at. A variable counts as listed when it
# is the first cell of a row of one of a document's tables.
set -euo pipefail
export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS=("$ROOT/docs/configuration.md" "$ROOT/docs/configuration.ja.md")
READERS='secure_getenv|getenv|env_str|env_size|impair_env'
LITERAL='"SRDSIM_[A-Z0-9_]+"'

fail=0
# Every call of a read helper outside comments, as file:line:text.
calls=$(grep -rnE --include='*.cc' --include='*.h' "\\b($READERS)[[:space:]]*\\(" "$ROOT/src" \
        | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|\*)' || true)
read_vars=$(grep -oE "\\b($READERS)[[:space:]]*\\([[:space:]]*$LITERAL" <<<"$calls" \
            | grep -oE 'SRDSIM_[A-Z0-9_]+' | sort -u || true)
# Calls that do not name a literal variable, other than the helpers' own definitions, the one
# getenv(name) each of them makes, and reads of variables outside SRDSIM_*.
opaque=$(grep -vE "\\b($READERS)[[:space:]]*\\([[:space:]]*(\"SRDSIM_[A-Z0-9_]+\"[[:space:]]*[,)]|\"[^S\"][^\"]*\")" <<<"$calls" \
         | grep -vE "\\b(env_str|env_size|impair_env)\\(const char[[:space:]]*\\*[[:space:]]*name" \
         | grep -vE 'std::getenv\(name\)' || true)
built=$(grep -rnE --include='*.cc' --include='*.h' '"SRDSIM_"|"SRDSIM_[A-Z0-9_]*%' "$ROOT/src" \
        | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|\*)' || true)
# A full name in quotes that no read helper takes, for instance one passed to a helper this script does
# not know.
quoted=$(grep -rhE --include='*.cc' --include='*.h' '"SRDSIM_[A-Z0-9_]+"' "$ROOT/src" \
         | grep -vE '^[[:space:]]*(//|\*)' | grep -oE '"SRDSIM_[A-Z0-9_]+"' | tr -d '"' | sort -u || true)
if [ -z "$read_vars" ]; then
    echo "found no variables read; the patterns in this script no longer match" >&2
    exit 1
fi
if [ -n "$opaque" ]; then
    echo "reads whose variable is not a literal name, which this check cannot follow:"; echo "$opaque"; fail=1
fi
if [ -n "$built" ]; then
    echo "variable names assembled at run time, which this check cannot follow:"; echo "$built"; fail=1
fi
unread=$(comm -23 <(echo "$quoted") <(echo "$read_vars"))
if [ -n "$unread" ]; then
    echo "names quoted in src/ that no known read helper takes:"; echo "$unread"; fail=1
fi
for doc in "${DOCS[@]}"; do
    name=${doc#"$ROOT/"}
    listed=$(awk -F'|' '/^[[:space:]]*\|/ {print $2}' "$doc" | grep -oE '`SRDSIM_[A-Z0-9_]+`' | tr -d '`' \
             | sort -u || true)
    if [ -z "$listed" ]; then echo "$name lists no variables" >&2; fail=1; continue; fi
    missing=$(comm -23 <(echo "$read_vars") <(echo "$listed"))
    stale=$(comm -13 <(echo "$read_vars") <(echo "$listed"))
    if [ -n "$missing" ]; then echo "read by src/ but not a row of $name:"; echo "$missing"; fail=1; fi
    if [ -n "$stale" ]; then echo "a row of $name but not read by src/:"; echo "$stale"; fail=1; fi
done
if [ "$fail" -eq 0 ]; then
    echo "docs/configuration.md and docs/configuration.ja.md each have a row for each of the $(wc -l <<<"$read_vars" | tr -d ' ') variables src/ reads"
fi
exit "$fail"

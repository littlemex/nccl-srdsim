#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails when a document, a workflow or a script names an upstream version that versions.env does not
# pin: a full commit hash, an NCCL release tag, an NCCL version in prose, or the NCCL wheel version.
# It checks these four kinds of reference only. A workflow's own action pins ("uses:" lines) are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
. ./versions.env
pinned_hashes=$(grep -oE '\b[0-9a-f]{40}\b' versions.env | sort -u)
fail=0
report() { echo "$1"; fail=1; }

# Relations inside versions.env that a reader relies on.
last=${NCCL_RELEASES##* }
[ "$NCCL_DEFAULT" = "$last" ] || report "versions.env: NCCL_DEFAULT=$NCCL_DEFAULT is not the newest of NCCL_RELEASES ($last)"
[ "v$NCCL_WHEEL" = "${NCCL_DEFAULT%-*}" ] || report "versions.env: NCCL_WHEEL=$NCCL_WHEEL is not the release of NCCL_DEFAULT=$NCCL_DEFAULT"
[[ " $CUDA_TOOLKITS " == *" $CUDA_DEFAULT "* ]] || report "versions.env: CUDA_DEFAULT=$CUDA_DEFAULT is not in CUDA_TOOLKITS"
for v in $(grep -oE '^NCCL_COMMIT_[A-Za-z0-9_]+' versions.env); do
    found=0
    for t in $NCCL_RELEASES; do [ "$v" = "NCCL_COMMIT_${t//[^A-Za-z0-9]/_}" ] && found=1; done
    [ "$found" -eq 1 ] || report "versions.env: $v belongs to no release in NCCL_RELEASES"
done
releases=" $NCCL_RELEASES "
files=$(git ls-files 'README.md' 'docs/*.md' '.github/workflows/*.yml' 'scripts/*' 'tests/gpu/*' 'ci/*' \
        | grep -v '^versions.env$')
for f in $files; do
    while IFS= read -r line; do
        n=${line%%:*}; text=${line#*:}
        # A workflow's action pin is not an upstream of this project.
        if [[ "$f" == .github/workflows/* ]] && [[ "$text" =~ ^[[:space:]]*(-[[:space:]]+)?uses: ]]; then continue; fi
        for h in $(grep -oE '\b[0-9a-f]{40}\b' <<<"$text" || true); do
            grep -qx "$h" <<<"$pinned_hashes" || report "$f:$n: commit $h is not pinned in versions.env"
        done
        for t in $(grep -oE '\bv2\.[0-9]+\.[0-9]+-[0-9]+\b' <<<"$text" || true); do
            [[ "$releases" == *" $t "* ]] || report "$f:$n: NCCL release $t is not in NCCL_RELEASES"
        done
        for v in $(grep -oE 'NCCL (version )?2\.[0-9]+\.[0-9]+' <<<"$text" | grep -oE '2\.[0-9]+\.[0-9]+' || true); do
            [[ "$releases" == *" v$v-"* ]] || report "$f:$n: NCCL $v is not in NCCL_RELEASES"
        done
        for v in $(grep -oE '(nvidia-nccl-cu[0-9]+==|NCCL_VERSION=)[0-9]+(\.[0-9]+)*' <<<"$text" | grep -oE '[0-9]+(\.[0-9]+)*$' || true); do
            [ "$v" = "$NCCL_WHEEL" ] || report "$f:$n: NCCL wheel $v differs from NCCL_WHEEL=$NCCL_WHEEL"
        done
    done < <(grep -nE '[0-9a-f]{40}|v2\.[0-9]+\.[0-9]+-[0-9]|NCCL (version )?2\.[0-9]|nvidia-nccl-cu[0-9]+==|NCCL_VERSION=[0-9]' "$f" || true)
done
[ "$fail" -eq 0 ] && echo "every version named in the documents, workflows and scripts is pinned in versions.env"
exit "$fail"

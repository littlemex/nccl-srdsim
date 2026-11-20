#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prints the GitHub Actions outputs CI builds its matrices from, read from versions.env:
#   nccl=<JSON list of {tag, commit, version_code}>   one entry per NCCL release
#   cuda=<JSON list of toolkit package suffixes>
#   nccl_default_tag, nccl_default_commit, cuda_default, cuda_keyring_sha256, gdrcopy_tag, gdrcopy_commit
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/versions.env"
commit_of() { local v="NCCL_COMMIT_${1//[^A-Za-z0-9]/_}"; [ -n "${!v:-}" ] || { echo "versions.env: no $v" >&2; exit 1; }; echo "${!v}"; }
code_of() {   # v2.32.6-1 -> 23206, as NCCL_VERSION_CODE
    local t=${1#v}; t=${t%%-*}; IFS=. read -r a b c <<<"$t"; printf '%d%02d%02d' "$a" "$b" "$c"
}
nccl="["; sep=""
for t in $NCCL_RELEASES; do
    nccl+="$sep{\"tag\":\"$t\",\"commit\":\"$(commit_of "$t")\",\"version_code\":\"$(code_of "$t")\"}"; sep=","
done
nccl+="]"
cuda="["; sep=""; for c in $CUDA_TOOLKITS; do cuda+="$sep\"$c\""; sep=","; done; cuda+="]"
echo "nccl=$nccl"
echo "cuda=$cuda"
echo "nccl_default_tag=$NCCL_DEFAULT"
echo "nccl_default_commit=$(commit_of "$NCCL_DEFAULT")"
echo "cuda_default=$CUDA_DEFAULT"
echo "cuda_keyring_sha256=$CUDA_KEYRING_SHA256"
echo "gdrcopy_tag=$GDRCOPY_TAG"
echo "gdrcopy_commit=$GDRCOPY_COMMIT"

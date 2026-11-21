#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# tests/gpu/recipe.sh [NRANKS] [SEEDS]
# From the repository root on a GPU host: installs the pinned NCCL wheel into .venv, fetches the pinned
# NCCL source for the plugin headers, builds the plugin and the two GPU tests, and runs tests/gpu/run.sh on each.
# Every version comes from versions.env; any failing step stops the recipe.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
. ./versions.env
command -v nvcc >/dev/null || { echo "nvcc is not on PATH (it installs to /usr/local/cuda/bin)" >&2; exit 2; }
python3 -m venv .venv
.venv/bin/pip install -q "nvidia-nccl-cu13==$NCCL_WHEEL"
NCCL_ROOT=$(.venv/bin/python -c 'import nvidia.nccl as m; print(list(m.__path__)[0])')
rm -rf .nccl-src
git clone --quiet --depth 1 --branch "$NCCL_DEFAULT" https://github.com/NVIDIA/nccl.git .nccl-src
test "$(git -C .nccl-src rev-parse HEAD)" = "$NCCL_DEFAULT_COMMIT" \
    || { echo ".nccl-src is not the pinned commit $NCCL_DEFAULT_COMMIT" >&2; exit 2; }
NCCL_SRC=$PWD/.nccl-src ./scripts/prepare_nccl_headers.sh "$PWD/.inc"
cmake -S . -B build -DSRDSIM_BUILD_TESTS=OFF -DSRDSIM_BUILD_PLUGIN=ON -DNCCL_PLUGIN_INCLUDE_DIRS="$PWD/.inc/nccl"
cmake --build build -j
for t in gin_signal_order gin_get; do
    nvcc -std=c++17 -arch=native -I "$NCCL_ROOT/include" "tests/gpu/$t.cu" -o "build/$t" \
        -L "$NCCL_ROOT/lib" -l:libnccl.so.2 -Xlinker -rpath -Xlinker "$NCCL_ROOT/lib"
done
./tests/gpu/run.sh "$PWD/build/libnccl-rma-srdsim.so" "$PWD/build/gin_signal_order" "${1:-3}" "${2:-5}"
# Gets are not reordered by the delivery modes (a reply is not scheduled), so two modes suffice, with each delivery: the
# default queued copies, with the control that counts a get as landed before its copy (it must fail every seed), and
# copies waited for one by one. Each run must have taken the delivery it names.
MODES="srd inorder control:srd" CONTROL_ENV="SRDSIM_GET_LANDED_EARLY=1" REQUIRE_LOG="CUDA copies, queued (asynchronous)" \
    CONTROL_EXPECT="the first wrong element was round ([0-9]+) block ([0-9]+) .*\(round [0-9]+'s value of rank [0-9]+ block \2\)" \
    ./tests/gpu/run.sh "$PWD/build/libnccl-rma-srdsim.so" "$PWD/build/gin_get" "${1:-3}" "${2:-5}"
SRDSIM_ASYNC_DELIVERY=0 MODES="srd inorder" REQUIRE_LOG="CUDA copies, each waited for" \
    ./tests/gpu/run.sh "$PWD/build/libnccl-rma-srdsim.so" "$PWD/build/gin_get" "${1:-3}" "${2:-5}"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# tests/gpu/recipe.sh [NRANKS] [SEEDS]
# From the repository root on a GPU host: installs the pinned NCCL wheel into .venv, fetches the pinned
# NCCL source for the plugin headers, builds the plugin and the GPU test, and runs tests/gpu/run.sh.
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
nvcc -std=c++17 -arch=native -I "$NCCL_ROOT/include" tests/gpu/gin_signal_order.cu -o build/gin_signal_order \
    -L "$NCCL_ROOT/lib" -l:libnccl.so.2 -Xlinker -rpath -Xlinker "$NCCL_ROOT/lib"
exec ./tests/gpu/run.sh "$PWD/build/libnccl-rma-srdsim.so" "$PWD/build/gin_signal_order" "${1:-3}" "${2:-5}"

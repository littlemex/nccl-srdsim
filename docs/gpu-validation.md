# Validating on a GPU

The host tests check the scheduler, transport and send queue without a GPU. What they cannot check is the plugin running under NCCL: that NCCL loads it, that GIN puts and signals reach its op table, that it writes device memory and signal words correctly, and that the delivery modes reach the GPU code under test. `tests/gpu/gin_signal_order.cu` checks that on one GPU.

## The test

The ranks form a ring. In every round, block b of each rank puts a buffer to block b of the next rank in chunks, and every chunk carries a weak signal on a signal id of its own. The receiver waits for each chunk's signal and checks that chunk as soon as the signal is visible, so a signal that overtakes its own put is caught even when the put lands a moment later. Every element encodes the round, the sending rank and its own index, so a chunk written to the wrong place, from the wrong sender or in the wrong round also counts as wrong. The receiver then returns a weak signal to the sender, which waits for it before refilling the slot; that signal only has to arrive, so the test asks the fabric for no ordering other than I1.

| Delivery | Required outcome |
|---|---|
| A mode that keeps I1 (`srd`, `inorder`, `peer-seq`, `reverse-ops`, `signal-last`, `defer-one`) | No wrong element, in every seed |
| `violate-i1` | Wrong elements in at least one seed and no crash or timeout in any. Only the plugin's scheduler can release a signal ahead of its data, so this also shows that the GIN traffic went through the plugin |

`tests/gpu/run.sh` runs the test in every mode over several seeds and checks these outcomes. It runs every rank as a separate process on GPU 0 with its own `NCCL_HOSTID`, so NCCL treats each rank as a separate node and reaches each peer only through GIN, and it starts its own MPS server so that the ranks' kernels run concurrently. Every wait in the kernel has a timeout (`WAIT_TIMEOUT`, 60 s by default) and every run has one (`RUN_TIMEOUT`, 180 s); when one rank fails the others are stopped. On failure the script prints the failing ranks' logs and keeps the work directory. CI compiles the test but cannot run it, so running it is a manual step.

## Requirements

- One NVIDIA GPU. The run on record used compute capability 12.0; other generations have not been tried
- The NVIDIA driver with its utilities, which provide `nvidia-cuda-mps-control`, and no MPS daemon already running
- The CUDA 13 toolkit, with `nvcc` on `PATH` (it installs to `/usr/local/cuda/bin`), since the NCCL wheel below is built for CUDA 13
- CMake 3.20 or later, a C++17 compiler, git, and Python 3 with `venv` (on Ubuntu, the `python3-venv` package)

## Run

```bash
export NCCL_VERSION=2.32.6
export PATH=/usr/local/cuda/bin:$PATH
git clone https://github.com/littlemex/nccl-srdsim.git
cd nccl-srdsim
python3 -m venv .venv
.venv/bin/pip install -q "nvidia-nccl-cu13==$NCCL_VERSION"
export NCCL_ROOT=$(.venv/bin/python -c 'import nvidia.nccl as m; print(list(m.__path__)[0])')
git clone --quiet --depth 1 --branch "v$NCCL_VERSION-1" https://github.com/NVIDIA/nccl.git .nccl-src
NCCL_SRC=$PWD/.nccl-src ./scripts/prepare_nccl_headers.sh "$PWD/.inc"
cmake -S . -B build -DSRDSIM_BUILD_TESTS=OFF -DSRDSIM_BUILD_PLUGIN=ON -DNCCL_PLUGIN_INCLUDE_DIRS="$PWD/.inc/nccl"
cmake --build build -j
nvcc -std=c++17 -arch=native -I "$NCCL_ROOT/include" tests/gpu/gin_signal_order.cu -o build/gin_signal_order -L "$NCCL_ROOT/lib" -l:libnccl.so.2 -Xlinker -rpath -Xlinker "$NCCL_ROOT/lib"
./tests/gpu/run.sh "$PWD/build/libnccl-rma-srdsim.so" "$PWD/build/gin_signal_order" 3 5
```

The last command prints one line per mode and exits 0 when every mode produced its required outcome.

## Result on record

On 2026-11-20, on a `g7.2xlarge` (one NVIDIA RTX PRO 4500 Blackwell Server Edition, compute capability 12.0, driver 595.112.07, CUDA 13.2) with NCCL 2.32.6: three ranks, five seeds per mode, 50 rounds of 65,536 elements in 16 chunks per block.

| Mode | Exit status per seed | Wrong elements reported by rank 0, per seed |
|---|---|---|
| `srd` | 0 0 0 0 0 | 0 in every seed |
| `inorder` | 0 0 0 0 0 | 0 in every seed |
| `peer-seq` | 0 0 0 0 0 | 0 in every seed |
| `reverse-ops` | 0 0 0 0 0 | 0 in every seed |
| `signal-last` | 0 0 0 0 0 | 0 in every seed |
| `defer-one` | 0 0 0 0 0 | 0 in every seed |
| `violate-i1` | 1 1 1 1 1 | 5,995,424; 6,746,688; 8,511,104; 11,900,576; 10,330,528 |

Each rank checks 13,107,200 elements per run (50 rounds of 4 blocks of 65,536), so under `violate-i1` between 46% and 91% of the elements rank 0 read had not yet landed when their signal had.

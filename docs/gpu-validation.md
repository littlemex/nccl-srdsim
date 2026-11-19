# Validating on a GPU

The host tests check the scheduler, transport and send queue without a GPU. What they cannot check is the plugin running under NCCL: that NCCL loads it, that GIN puts and signals reach its op table, that it writes device memory and signal words correctly, and that the delivery modes reach the GPU code under test. `tests/gpu/gin_signal_order.cu` checks that on one GPU.

## The test

The ranks form a ring of thread blocks. In every round each block puts a buffer to the next block in chunks, each chunk carrying a weak signal increment. The receiver waits until the signal has counted every chunk of the round, checks every element, and returns a strong signal so that the sender may reuse the slot. The only thing that orders the data before the check is invariant I1, so:

| Delivery | Required outcome |
|---|---|
| A mode that keeps I1 (`srd`, `inorder`, `peer-seq`, `reverse-ops`, `signal-last`, `defer-one`) | No wrong element, in every seed |
| `violate-i1` | Wrong elements in at least one seed. Only the plugin's scheduler can release a signal ahead of its data, so this also shows that the GIN traffic went through the plugin |

`tests/gpu/run.sh` runs the test in every mode over several seeds and checks these outcomes. It runs every rank as a separate process on GPU 0 with its own `NCCL_HOSTID`, so NCCL treats each rank as a separate node and reaches each peer only through GIN, and it starts an MPS server so that the ranks' kernels run at the same time.

## Requirements

- One NVIDIA GPU with compute capability 9.0 or later, its driver, and the CUDA toolkit (`nvcc`)
- CMake 3.20 or later, a C++17 compiler, git and Python 3 with `venv`

## Run

```bash
export NCCL_VERSION=2.32.6
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

# Validating on a GPU

**English** | [日本語](gpu-validation.ja.md)

This page describes the one test of nccl-srdsim that needs a GPU. nccl-srdsim is a plugin that NCCL loads to carry the data GPU kernels send each other; it delivers that data in a different order on every seed, the way an unordered but reliable network such as AWS SRD may. The test checks that, under real NCCL, code written against NCCL's ordering guarantee still sees correct data, and that code which relied on more than the guarantee would be caught.

The vocabulary, once. NCCL's GIN (GPU-Initiated Networking) device API lets a CUDA kernel send data itself. A *put* copies a range of the sender's GPU memory into the receiver's GPU memory. A *signal* is a 64-bit counter in the receiver's memory that a put can increment once its data is written, and that the receiver's kernel polls to learn that the data is there. A *weak* signal promises only that: it becomes visible after the data of its own put has landed, and promises nothing about other puts. That promise is invariant I1 (see the [README](../README.md#delivery-modes)). A *delivery mode* is the rule by which the plugin reorders deliveries, and a *seed* fixes the plugin's random choices within that rule; the GPU's own timing still differs from one execution to the next.

`tests/gpu/gin_signal_order.cu` is a small CUDA program whose result is correct only if I1 holds, and `tests/gpu/run.sh` runs it once per delivery mode and seed. Together they answer three questions the host tests cannot:

1. Does NCCL load the plugin and send a kernel's puts and signals through it?
2. Does the plugin write GPU memory and signal counters correctly?
3. When a delivery breaks I1, does a kernel see wrong data?

![Two panels. Left, the host tests that run in CI without a GPU or NCCL: the scheduler, which decides the order operations are delivered in, the transport, which carries data between processes, and the send queue, which holds data copied out of the GPU; the plugin under NCCL and its writes into GPU memory never run there. Right, what the GPU test adds, from top to bottom: a GPU kernel issuing GIN puts with weak signals, NCCL's GIN proxy backend, this plugin loaded through NCCL_RMA_PLUGIN, and GPU memory holding the data and the signal counters.](images/gpu-why.svg)

Six of the seven delivery modes keep I1, so under them no element of the transferred data may be wrong. The seventh, `violate-i1`, may release a signal ahead of its data, on purpose, so at least one seed must find mismatched elements. Only the plugin can release a signal ahead of its data, so mismatches under `violate-i1` also show that the kernel's traffic went through the plugin.

| Mode | Required outcome |
|---|---|
| A mode that keeps I1 (`srd`, `inorder`, `peer-seq`, `reverse-ops`, `signal-last`, `defer-one`) | No mismatched element, in every seed |
| `violate-i1` | Mismatched elements in at least one seed, and no crash or timeout in any |

CI compiles the test (the `gpu-test-compile` job) but has no GPU, so running it is a manual step.

## How the test works

The test passes data around a ring of ranks and checks every chunk as soon as its signal says the chunk has landed. The parts and what each does:

| Part | Does |
|---|---|
| Rank | One process on GPU 0, launched by `run.sh` with its own `NCCL_HOSTID`, which is meant to make NCCL treat each rank as a separate machine and route its puts and signals through the plugin rather than copying between the processes directly. The test's evidence that the traffic went through the plugin is the mismatches under `violate-i1`. Every recorded run used separate `NCCL_HOSTID`s; one shared value has not been tried |
| MPS server | CUDA's Multi-Process Service, started by `run.sh`; runs the ranks' kernels on the GPU at the same time instead of taking turns |
| Block | One of the 4 thread blocks of a rank's kernel. Block b of rank r sends to block b of rank r + 1, and the last rank sends to rank 0; every block is at once a sender to the next rank and a receiver from the previous one |
| Slot | A block's buffer of 65,536 32-bit elements: one send slot and one receive slot per block |
| Chunk | One put of 4,096 elements; a slot travels as 16 chunks, each with its own signal counter |
| Round | One full exchange of a slot. A seed runs 50 rounds by default |
| Plugin | Carries every put and signal between the ranks, in the order of the delivery mode |

![Top: three rank processes on GPU 0, shared through one MPS server, each with its own NCCL_HOSTID; rank 0 puts to rank 1, rank 1 to rank 2, rank 2 to rank 0. Bottom: one block in one round. (1) Block b of rank r puts its send slot of 65,536 elements to block b of rank r + 1 as 16 chunks of 4,096 elements, each with a weak signal of its own id. (2) The receiver checks each chunk once its signal is visible. (3) It returns a weak signal saying the slot is free. (4) The sender flushes; in the next round it waits for that signal, refills its slot and puts again.](images/gpu-ring.svg)

One round of one block, numbered as in the figure:

1. The sending block fills its send slot and puts it to the receiving block's slot as 16 chunks. Every chunk carries a weak signal with a signal id of its own.
2. The receiving block checks each chunk once that chunk's signal is visible (next section).
3. When all 16 chunks are checked, the receiver sends a weak signal back to the sender, meaning that its receive slot may be overwritten. The test does not care when this signal arrives relative to anything else, so I1 is the only ordering it depends on.
4. After sending its own free signal upstream, each block calls flush, which NCCL defines as waiting until the puts it issued in step 1 have read their source, so that the send slot may be refilled. Independently, at the start of the next round, the block waits for the downstream receiver's signal from step 3 before it refills its send slot.

The kernel's three waits, for a free slot, for a chunk's signal and for a flush, have a time limit (60 s by default); a rank whose wait runs out stops and exits with status 3 once its kernel has returned.

### Checking each chunk as its signal appears

The way the receiver checks is what makes a broken I1 detectable: it reads a chunk soon after the chunk's signal says the data is there, so a signal that came too early is caught while its data is still missing. Thread 0 of the receiving block scans the chunks it has not checked yet, from chunk 0 up, and picks the first whose signal is visible; then all threads of the block compare that chunk with the values they expect, and the scan starts again. A chunk whose signal is visible never waits for a signal that has not arrived; it waits only for the checks of lower-numbered chunks whose signals are also visible.

![Two timelines of one round at the receiver, each with three rows: data lands, signal visible, block checks. Top, a mode that keeps I1: for chunks C2, C0 and C1, the data lands before its signal s2, s0, s1, and each check, done after its signal, passes. Bottom, violate-i1: signal s1 becomes visible before chunk C1's data lands, so the check of C1 fails and its elements count as mismatched; C2 and C0 pass. Three of the 16 chunks are shown; chunks may land in any order, and only a signal ahead of its own data is forbidden.](images/gpu-check-order.svg)

Chunks may land in any order; only a signal ahead of its own data breaks I1. A put whose data lands between its signal becoming visible and the chunk being read, a gap that can last several chunk checks, is not caught, so a seed under `violate-i1` finds some of the early signals, not all of them. That is why `violate-i1` is required to find mismatches in at least one seed rather than in every seed.

### What each element holds

Each element is computed from the round, the sending rank, the block and the element's index, so a chunk from the wrong round, rank, block or offset counts as mismatched too, not only data that has not landed. The receive slots are not initialised, so in round 0 a slot holds whatever the buffer held when NCCL allocated it. The sender writes, and the receiver expects, for element `i` of block `b` in round `r` from rank `s`:

```text
((r + 1) * 0x9E3779B1) ^ (s * 0x85EBCA6B) ^ (b * 0xC2B2AE35) ^ i        (32-bit unsigned)
```

The receiver computes this from the round it is checking, its upstream rank, its block and the index, and counts every element that differs. With `r + 1`, round 0 is not multiplied by zero.

![Four fields combined by XOR into the expected 32-bit value: round + 1 times 0x9E3779B1, the sender rank times 0x85EBCA6B, the block times 0xC2B2AE35, and the index as is. Below, the misdeliveries this catches: data not yet landed, with the previous round's value left in the slot (in round 0, the buffer's initial contents), differs in the round; a chunk from the wrong rank differs in the sender; a chunk written into another block's slot differs in the block; a chunk at the wrong offset differs in the index.](images/gpu-encoding.svg)

Each multiplier is odd, so multiplying by it modulo 2^32 never maps two different values of one field to the same result: an element that differs from the expected one in exactly one of the four fields always holds a different value. An element that differs in two or more fields could, in principle, cancel out.

## The driver: tests/gpu/run.sh

`run.sh` runs the test in every mode over several seeds, decides from the ranks' exit statuses whether each mode produced its required outcome, and makes sure that one broken rank cannot hang the whole run. It is called as `tests/gpu/run.sh PLUGIN_SO TEST_BINARY [NRANKS] [SEEDS]`, with 3 ranks and 5 seeds by default.

![The control flow of run.sh. (1) Start its own MPS server, refusing to start next to another. Then, for each mode and each seed: (2) launch every rank, each with its own NCCL_HOSTID, the same mode and seed, and under a timeout; (3) wait for each rank to exit: on exit 0 or 1 keep waiting, on exit 3 give the other ranks 10 s and then stop them, on any other exit (a setup error, a crash, or the run time limit) stop them at once; (4) take the seed's status from the first exit other than 0 or 1, otherwise 1 if any rank exited 1, otherwise 0, and restart MPS after a status of 3 or more or after a rank was stopped by timeout or a signal (124 or more). (5) Per mode: the modes that keep I1 pass when every seed is 0; violate-i1 passes when every seed is 0 or 1 and at least one is 1.](images/gpu-run-flow.svg)

Numbered as in the figure:

1. `run.sh` starts an MPS server of its own in a temporary directory. If another MPS daemon is already running on the machine it refuses to start.
2. For each mode and seed it launches every rank with its own `NCCL_HOSTID`, the plugin (`NCCL_RMA_PLUGIN`, and `NCCL_GIN_TYPE=2`, which selects the GIN backend that calls an RMA plugin), the mode and the seed, each under the coreutils `timeout` command with `RUN_TIMEOUT`.
3. It waits for the ranks. A rank that exits 0 or 1 has finished normally. A rank that exits 3 has timed out, and the other ranks are likely to time out too, so they get 10 seconds to do so and write their logs before they are stopped. A rank that exits with anything else had a setup error, crashed, or was stopped by `RUN_TIMEOUT`, and the other ranks would only wait for it until their own time limit, so they are stopped at once. `run.sh` polls every 0.2 s, in rank order, and the first such exit it notices decides how the other ranks are stopped.
4. The seed's status is that first exit other than 0 or 1; otherwise it is 1 if any rank exited 1, and 0 if none did. A rank stopped in the middle of its kernel may leave the MPS server holding its context, so after a status of 3 or more, or after a rank was stopped by `timeout` or by a signal (an exit status of 124 or more), `run.sh` restarts the MPS server.
5. When all seeds of a mode are done, the mode passes if it produced its required outcome: for a mode that keeps I1, every seed's status is 0; for `violate-i1`, every seed's status is 0 or 1 and at least one is 1. `run.sh` prints the mode's line ending in `PASS` or `FAIL`, then, for each seed, rank 0's result line if rank 0 wrote one; that line reports the mismatched elements as `wrong elements`. When the mode fails, it also prints the last 5 lines of every rank's log for each seed that failed (none when `violate-i1` fails only because no seed found a mismatch). It exits 0 when every mode passed. It keeps its work directory, with every rank's log, only when it fails; after a passing run the logs are deleted.

The test program's exit status, which the verdict is built from:

| Exit status | Meaning |
|---|---|
| 0 | No element was mismatched |
| 1 | Some elements were mismatched |
| 2 | Bad arguments, no unique-id file from rank 0 within 60 s, or a failed CUDA or NCCL call, including those that shut NCCL down at the end |
| 3 | A wait in the kernel (for a free slot, a chunk's signal or a flush) timed out. The rank exits without shutting NCCL down, because operations may still be pending and the other ranks may be gone |

`run.sh` reads these environment variables:

| Variable | Default | Effect |
|---|---|---|
| `RUN_TIMEOUT` | `180` | Seconds a rank may run before `timeout` stops it. Also ends a put or signal call that never returns |
| `WAIT_TIMEOUT` | `60` | Seconds the kernel waits for one signal or one flush before the rank gives up with status 3 |
| `ROUNDS` | `50` | Rounds per seed |
| `MODES` | all seven | The delivery modes to run, separated by spaces |

## Requirements

- One NVIDIA GPU. The run on record used compute capability 12.0; other generations have not been tried
- The NVIDIA driver with its utilities, which provide `nvidia-cuda-mps-control`, and no MPS daemon already running
- The CUDA 13 toolkit, with `nvcc` on `PATH` (it installs to `/usr/local/cuda/bin`), since the NCCL wheel below is built for CUDA 13
- CMake 3.20 or later, a C++17 compiler, git, and Python 3 with `venv` (on Ubuntu, the `python3-venv` package)

## Run

The commands below install the NCCL 2.32.6 library from its Python wheel, fetch the NCCL source for the plugin's build headers, build the plugin, compile the test against the wheel's NCCL, and run every mode with 3 ranks and 5 seeds. The last step takes about 75 seconds.

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

The last command prints, for each mode, a line ending in `PASS` or `FAIL` followed by rank 0's count of mismatched elements for each seed, and exits 0 when every mode produced its required outcome.

## Result on record

All seven modes produced their required outcome in each of four runs of `run.sh`.

![For each of the seven modes, the exit status of each of the five seeds, the same in every run: 0 in every seed of the six modes that keep I1, which means no mismatched element, and 1 in every seed of violate-i1, which means mismatched elements were found; every mode met its required outcome. Below, the range of mismatched elements rank 0 found per seed under violate-i1: from 3,157,632 (24%) to 11,900,672 (91%) of the 13,107,200 elements it checks per seed.](images/gpu-result.svg)

The runs used the plugin and the test as they are at commit `7925c81` (measured at commit `ff11cc9`, before a rebase that changed only documents; no source or test differs), on a `g7.2xlarge` (one NVIDIA RTX PRO 4500 Blackwell Server Edition, compute capability 12.0, driver 595.112.07) running the Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261117, with NCCL 2.32.6 from the wheel above and the CUDA 13 toolkit on the image. The commands under Run were executed four times: once exactly as written above, and three more times with the same build and the same `tests/gpu/run.sh PLUGIN TEST 3 5` command. Each run used three ranks, five seeds per mode, and 50 rounds of 65,536 elements in 16 chunks per block, and took about 75 seconds.

| Mode | Exit status per seed, in every run | Mismatched elements reported by rank 0 |
|---|---|---|
| `srd`, `inorder`, `peer-seq`, `reverse-ops`, `signal-last`, `defer-one` | 0 0 0 0 0 | 0 in every seed of every run |
| `violate-i1` | 1 1 1 1 1 | Between 3,157,632 and 11,900,672 per seed, over seeds 1 to 5 of the four runs |

Each rank checks 13,107,200 elements per seed (50 rounds of 4 blocks of 65,536), so under `violate-i1` between 24% and 91% of the elements rank 0 checked did not hold the expected value.

The timeout path was run once at the same commit, with `MODES=srd ROUNDS=20000 WAIT_TIMEOUT=5` and two seeds, freezing rank 1 with `SIGSTOP` 20 seconds into seed 1. Ranks 0 and 2 logged that their four blocks timed out after 5 s, the first in round index 5,231 and 5,237 of 20,000, waiting for a free slot and for a chunk's signal respectively; seed 1 was reported as 3, the MPS server was restarted, seed 2 exited 0, and `run.sh` exited 1 with the logs kept.

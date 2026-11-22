# nccl-srdsim

An NCCL RMA plugin that simulates the delivery semantics of an unordered, reliable transport such as
AWS's Scalable Reliable Datagram (SRD), on GPU machines with no EFA device. It lets code built on NCCL's
GIN (GPU-initiated networking) device API be tested against the orderings such a fabric is allowed to
produce: operations that complete out of order, operations torn into chunks that land in any order, and
signals that may only be relied on once their own payload has landed.

Its purpose is correctness, not performance. It stages payloads through host memory and makes no claim
about bandwidth or latency.

## What it covers

NCCL's GIN proxy backend (`NCCL_GIN_TYPE=2`) routes every device-initiated put, signal and get through the host RMA plugin interface. This plugin implements that interface, so anything whose remote writes travel through it is in scope, whatever library issued them:

| In scope | Out of scope |
|---|---|
| Any GIN user's puts, signalled puts, bare signals and gets through the proxy backend | NCCL's ordinary collectives, which use the network plugin interface |
| The ordering assumptions such code makes about the fabric | Device-side GIN backends (GDAKI, EFA-GDA), where the GPU drives the NIC and no host op table is on the path |
| Contract violations injected on purpose, to show a test can fail | Libraries built on NVSHMEM |
| | Wire encoding, doorbells, PCIe ordering, and anything measured in seconds |

Code that reaches EFA through libfabric rather than NCCL's RMA interface, such as aws-ofi-nccl and NIXL's libfabric backend, is served by a second front end, a libfabric provider named `efa`. It is being built; what it does so far is in [docs/libfabric-provider.md](docs/libfabric-provider.md).

## Delivery modes

The default mode keeps the weak-signal guarantee of NCCL's [RMA plugin specification](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/plugins/rma/README.md#L92-L97), which is what an unordered fabric can provide: a signal becomes visible only after every byte of its own put has landed, with no guarantee about earlier puts or signals. This documentation calls that rule invariant I1. Nothing else is ordered: puts complete in any order relative to each other, and the chunks of one put land in any order.

`SRDSIM_MODE` selects the schedule:

| Mode | What it does |
|---|---|
| `srd` (default) | Splits each operation into chunks and releases them in a random order across operations; a signal is held until every byte of its own operation has landed |
| `inorder` | Delivers in submission order. The control: when code fails here, the failure is in the code under test, not in the schedule |
| `peer-seq` | Keeps submission order per peer; chunks within an operation may still be reordered. This is stronger than the contract. It matches how the GIN implementation in aws-ofi-nccl retires received operations, in per-peer sequence order ([`rx_consumed`](https://github.com/aws/aws-ofi-nccl/blob/2475c4c736907e68ebeadb3848f31231ee4a44d0/include/rdma/gin/nccl_ofi_gin_types.h#L86-L89) at that commit), so code that passes here but fails under `srd` relies on more than the contract, though not on more than that implementation |
| `reverse-ops`, `signal-last`, `defer-one` | Legal schedules built to break one assumption each: newest operation first; every payload before any signal; one operation held back until all others have landed |
| `violate-i1` | May release a signal before its own put has fully landed, breaking I1 on purpose. Code that relies on I1, as correct code does, should fail under it for some seed; a test that never fails here does not check what it reads |

`SRDSIM_SEED` makes a schedule reproducible.

## Requirements

| To | Needs |
|---|---|
| Run the host tests | A C++17 compiler and CMake 3.20 or later |
| Build the plugin | The plugin headers of NCCL 2.31.2 to 2.32.6 (from an NCCL source checkout, see below) and the CUDA toolkit, whose headers `nccl.h` includes; NVRTC for delivery into device memory; optionally GDRCopy. Built in CI against the CUDA toolkits in `versions.env` |
| Run the plugin | An NVIDIA GPU, an NCCL release from the table in [NCCL versions](#nccl-versions), and the CUDA driver. A build with the CUDA delivery path also loads the NVRTC runtime of the CUDA version it was built with, and a GDRCopy build loads `libgdrapi.so`; if either cannot be found, the plugin does not load |

## NCCL versions

NCCL tries `ncclRmaPlugin_vN` from the newest version it knows downwards and uses the first the plugin exports, so one build serves several NCCL releases. Built against the newest supported headers, the plugin exports `ncclRmaPlugin_v14`, `ncclRmaPlugin_v15` and `ncclRmaPlugin_v16`:

| NCCL release | First table it tries that this plugin exports |
|---|---|
| 2.31.2 | v15 |
| 2.32.3 | v16 |
| 2.32.6 | v16 (2.32.6 also defines v17, which the plugin does not export yet: [#13](https://github.com/littlemex/nccl-srdsim/issues/13)) |

Every upstream version the repository builds and tests against (NCCL releases and their commits, CUDA toolkits, GDRCopy, the reference aws-ofi-nccl commit) is pinned in [`versions.env`](versions.env), which CI and the GPU recipe read. CI fails when a document, workflow or script names a full commit hash, an NCCL release tag, an NCCL version written as `NCCL x.y.z`, or an NCCL wheel version that `versions.env` does not pin.

## Build

Host tests (no GPU, no NCCL):

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The plugin, against the CUDA toolkit and the plugin headers of the NCCL release it will be loaded by. The plugin headers (`nccl_rma.h`) are not shipped in NCCL's packages, so they come from an NCCL source checkout of the same version:

```bash
git clone --depth 1 --branch v2.32.6-1 https://github.com/NVIDIA/nccl.git /tmp/nccl-src
NCCL_SRC=/tmp/nccl-src ./scripts/prepare_nccl_headers.sh /tmp/srdsim-inc
cmake -S . -B build -DSRDSIM_BUILD_PLUGIN=ON -DNCCL_PLUGIN_INCLUDE_DIRS=/tmp/srdsim-inc/nccl
cmake --build build -j
```

The script also writes `/tmp/srdsim-inc/cuda-stub`, declaration-only CUDA stand-ins used to type-check the plugin without CUDA; a real build must not add it to the include path.

| CMake option | Default | Effect |
|---|---|---|
| `SRDSIM_BUILD_TESTS` | `ON` | Build the host tests |
| `SRDSIM_BUILD_PLUGIN` | `OFF` | Build `libnccl-rma-srdsim.so` |
| `SRDSIM_WITH_CUDA` | `ON` | Deliver into device memory through the CUDA driver API. Without it the plugin handles host memory only |
| `SRDSIM_WITH_GDRCOPY` | `OFF` | Also deliver into device memory through GDRCopy BAR mappings when `/dev/gdrdrv` is present. Needs `SRDSIM_WITH_CUDA` |
| `SRDSIM_SANITIZER` | empty | Build with a sanitizer: `address`, `undefined` or `thread`. Under `ctest`, which sets the sanitizer options itself (overriding any in the environment), a finding fails the test. The sanitizers of GCC 13 need `sudo sysctl -w vm.mmap_rnd_bits=28` on hosts with more ASLR entropy, such as Ubuntu 24.04; ThreadSanitizer fails at startup without it |

## Use

```bash
export NCCL_RMA_PLUGIN=$PWD/build/libnccl-rma-srdsim.so
export NCCL_GIN_TYPE=2
export SRDSIM_MODE=srd SRDSIM_SEED=1
```

Every `SRDSIM_*` setting, with its default, is listed in [docs/configuration.md](docs/configuration.md). CI builds the plugin in every configuration and loads the build without the CUDA delivery path with `dlopen`, looking up its op tables as NCCL does; it does not run NCCL. A run under NCCL on a GPU is being validated in [#8](https://github.com/littlemex/nccl-srdsim/issues/8).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for how to report a security issue.

## License

This project is licensed under the Apache-2.0 License. See [LICENSE](LICENSE).

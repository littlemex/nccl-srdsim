# nccl-srdsim

An NCCL RMA plugin that simulates the delivery semantics of an unordered, reliable transport such as
AWS's Scalable Reliable Datagram (SRD), on any machine, with no EFA device. It lets code built on NCCL's
GIN (GPU-initiated networking) device API be tested against the orderings such a fabric is allowed to
produce: operations that complete out of order, operations torn into chunks that land in any order, and
signals that may only be relied on once their own payload has landed.

Its purpose is correctness, not performance. It stages payloads through host memory and makes no claim
about bandwidth or latency.

## What it covers

NCCL's GIN proxy backend (`NCCL_GIN_TYPE=2`) routes every device-initiated put, signal and get through
the host RMA plugin interface. This plugin implements that interface (it exports `ncclRmaPlugin_v14`, and `ncclRmaPlugin_v15` when built against headers that define it; CI covers NCCL 2.31.2, 2.32.3 and 2.32.6), so anything
whose remote writes travel through it is in scope, whatever library issued them:

| In scope | Out of scope |
|---|---|
| Any GIN user's puts, signalled puts, bare signals and gets through the proxy backend | NCCL's ordinary collectives, which use the network plugin interface |
| The ordering assumptions such code makes about the fabric | Device-side GIN backends (GDAKI, EFA-GDA), where the GPU drives the NIC and no host op table is on the path |
| Contract violations injected on purpose, to show a test can fail | Libraries built on NVSHMEM |
| | Wire encoding, doorbells, PCIe ordering, and anything measured in seconds |

## Delivery modes

The contract the default mode keeps is the one SRD gives for a signalled put: the signal becomes visible only after every byte of its own put has landed. Nothing else is ordered: puts complete in any order relative to each other, and the chunks of one put land in any order. The documentation calls the signal rule invariant I1.

`SRDSIM_MODE` selects the schedule:

| Mode | What it does |
|---|---|
| `srd` (default) | Splits each operation into chunks and releases them in a random order across operations; a signal is held until every byte of its own operation has landed |
| `inorder` | Delivers in submission order. The control: when code fails here, the failure is in the code under test, not in the schedule |
| `peer-seq` | Keeps submission order per peer; chunks within an operation may still be reordered. This is stronger than the contract. It matches how the GIN implementation in aws-ofi-nccl retires received operations, in per-peer sequence order ([`rx_consumed`](https://github.com/aws/aws-ofi-nccl/blob/2475c4c736907e68ebeadb3848f31231ee4a44d0/include/rdma/gin/nccl_ofi_gin_types.h#L86-L89) at that commit), so code that passes here but fails under `srd` relies on more than the contract, though not on more than that implementation |
| `reverse-ops`, `signal-last`, `defer-one` | Legal schedules built to break one assumption each: newest operation first; every payload before any signal; one operation held back until all others have landed |
| `violate-i1` | May release a signal before its own put has fully landed, breaking I1 on purpose. Code that relies on I1, as correct code does, should fail under it for some seed; a test that never fails here does not check what it reads |

`SRDSIM_SEED` makes a schedule reproducible.

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

## Use

```bash
export NCCL_RMA_PLUGIN=$PWD/build/libnccl-rma-srdsim.so
export NCCL_GIN_TYPE=2
export SRDSIM_MODE=srd SRDSIM_SEED=1
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for how to report a security issue.

## License

This project is licensed under the Apache-2.0 License. See [LICENSE](LICENSE).

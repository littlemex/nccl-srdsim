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
the host RMA plugin interface (`ncclRmaPlugin_v14`). This plugin implements that interface, so anything
whose remote writes travel through it is in scope, whatever library issued them:

| In scope | Out of scope |
|---|---|
| Any GIN user's puts, signalled puts, bare signals and gets through the proxy backend | NCCL's ordinary collectives, which use the network plugin interface |
| The ordering assumptions such code makes about the fabric | Device-side GIN backends (GDAKI, EFA-GDA), where the GPU drives the NIC and no host op table is on the path |
| Contract violations injected on purpose, to show a test can fail | Libraries built on NVSHMEM |
| | Wire encoding, doorbells, PCIe ordering, and anything measured in seconds |

## Delivery modes

`SRDSIM_MODE` selects the schedule:

| Mode | What it does |
|---|---|
| `srd` (default) | Splits each operation into chunks and releases them in a random order across operations; a signal is held until every byte of its own operation has landed |
| `inorder` | Delivers in submission order. The control: a failure here is in the code under test, not in the schedule |
| `peer-seq` | Keeps submission order per peer; chunks within an operation may still be reordered. Stronger than the contract, and what the AWS OFI GIN plugin provides, so it separates relying on more than the contract from relying on more than that plugin |
| `reverse-ops`, `signal-last`, `defer-one` | Legal schedules built to break one assumption each: newest operation first; every payload before any signal; one operation held back until all others have landed |
| `violate-i1` | Releases a signal before its own payload has fully landed. A correct test must fail under it |

`SRDSIM_SEED` makes a schedule reproducible.

## Build

Host tests (no GPU, no NCCL):

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The plugin, against an installed NCCL and the CUDA toolkit. The plugin headers (`nccl_rma.h`) are not
shipped in NCCL's Python wheels, so point at an NCCL source checkout of the same version as well:

```bash
cmake -S . -B build -DSRDSIM_BUILD_PLUGIN=ON \
  -DNCCL_PLUGIN_INCLUDE_DIRS="$NCCL_HOME/include;$NCCL_SRC/src/include/plugin;$NCCL_SRC/src/include"
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

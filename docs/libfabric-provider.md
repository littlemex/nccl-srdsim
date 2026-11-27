# The libfabric provider

**English** | [日本語](libfabric-provider.ja.md)

`libsrdsim-fi.so` is a libfabric provider that registers under the name `efa`. Code that reaches EFA through libfabric, such as aws-ofi-nccl and NIXL's libfabric backend, takes its EFA branches only when the provider is called `efa`; this provider lets that code run between processes on one host, on an EFA instance without using its devices or on a host with no EFA at all, with the delivery rules of SRD. The design and its open questions are in [issue #75](https://github.com/littlemex/nccl-srdsim/issues/75).

## Status

| Part | State |
|---|---|
| Registration as `efa`, selection, `fi_getinfo` | Done |
| Fabric, domain, address vector, completion queue and endpoint objects; the EFA endpoint options | Done |
| Sends and receives (untagged and tagged), RMA writes, writes with CQ data and reads between processes, with SRD's orderings | Done (`src/fi/srdsim_xfer.c`) |
| Memory registration of host and CUDA memory, with provider keys | Done, through libfabric's registration map; CUDA needs libfabric configured `--with-cuda` |
| NCCL over aws-ofi-nccl on one GPU host, with its send/recv and RDMA protocols | Runs: two ranks on one GPU, `all_reduce` correct under `srd` and `inorder`. On a host with a real EFA device, the same job is correct through the device and through this provider, and the cost of each is below |
| NIXL's libfabric backend on one host | Runs: NIXL's own libfabric unit tests pass against this provider (backend, source-address resolution, rail refcount), its topology discovery reads the emulated NICs' PCIe addresses and link speed, and two agents move real data through it end to end |

## Building

The provider is built the way libfabric builds its own providers as loadable libraries: `scripts/build_fi_provider.sh` adds `src/fi` to a libfabric source tree as `prov/srdsim`, builds that libfabric with the provider as a DSO, and installs it under a prefix. The provider uses libfabric's own utility layer for hint matching, the fabric, domain, completion queue and address vector objects, and memory registration, so it answers those calls exactly as libfabric's other providers do. It is moved to `PREFIX/lib/srdsim/`, outside libfabric's default provider directory. CI builds it against the libfabric release pinned in `versions.env`.

```bash
git clone --quiet --depth 1 --branch v2.7.0 https://github.com/ofiwg/libfabric.git libfabric-src
./scripts/build_fi_provider.sh libfabric-src "$HOME/srdsim-libfabric"
```

The script needs autoconf, automake, libtool and a C compiler. Configure options after the prefix are passed to libfabric's `configure`. The provider has been run only with the libfabric it was built with.

## Selecting it

The provider is used only by a process that names it, in the preferred form of `FI_PROVIDER_PATH`: a `+` followed by the library's path. libfabric then gives it priority over any other provider called `efa`, including the real one, so on an EFA instance the process sees only the emulated device. It is never installed in libfabric's default provider directory, and a process that does not name it gets the real provider.

```bash
export LD_LIBRARY_PATH="$HOME/srdsim-libfabric/lib"
export FI_PROVIDER_PATH="+$HOME/srdsim-libfabric/lib/srdsim/libsrdsim-fi.so"
"$HOME/srdsim-libfabric/bin/fi_info" -p efa
```

`fi_info` then lists one domain, `srdsim0-rdm`.

On a host that has a real EFA device, with libfabric built `--enable-efa=yes` so that the real provider is there too, both halves of that were measured: not named, `fi_info -p efa` answers only for the device (`rdmap88s0-rdm`, on the `efa` and `efa-direct` fabrics) and this provider is not loaded; named, this provider takes the name (`srdsim0-rdm`, `srdsim1-rdm`); and after a full test run the device still answers on all of its fabrics.

## What it reports

| Attribute | Value | Why |
|---|---|---|
| Endpoint | `FI_EP_RDM`, protocol `FI_PROTO_EFA`, address format `FI_ADDR_EFA`, 32-byte addresses | As the real RDM endpoint |
| Ordering | `FI_ORDER_SAS` for sends and receives, nothing for RMA | The real provider reorders two-sided messages back in software; SRD itself orders nothing, and hints that ask for an RMA ordering are refused |
| Registration | `FI_MR_VIRT_ADDR`, `FI_MR_ALLOCATED` and `FI_MR_PROV_KEY` for RMA, `FI_MR_HMEM` with `FI_HMEM`, and `FI_MR_LOCAL` when the hints offer it; 4-byte keys | As the real RDM endpoint, which uses a consumer's registrations when it offers them |
| Capabilities | Messages, tagged messages and RMA, with `FI_HMEM` only when asked for; no atomics | The real provider emulates atomics in software, and nothing this provider serves uses them |
| CQ data | 4 bytes | What `fi_writedata` carries on EFA |
| Progress | Manual, for data and control | As the real provider; hints asking for automatic progress are refused |
| Threading | `FI_THREAD_SAFE` when the hints ask for nothing in particular, and otherwise whatever they ask for -- but the domain's objects lock as though it were `FI_THREAD_SAFE` whatever the answer | libfabric lets the hints decide the reported model (`ofi_alter_info`), and its utility layer then reads that one field to choose locks. Asking for `FI_THREAD_COMPLETION` or `FI_THREAD_DOMAIN` disarms the one that matters here, the completion ring (`ofi_cq_init`), which both the writer and the reader of that ring take. The endpoint's own lock is disarmed too (`ofi_endpoint_init`, for anything other than `FI_THREAD_SAFE`) but is inert today: nothing in libfabric or this provider ever takes it, only initialises and destroys it. The endpoint-list and address-vector locks are not affected at all, because those become no-ops only under `FI_PROGRESS_CONTROL_UNIFIED` and this provider reports manual control progress. In those models a consumer owes libfabric its own serialisation per queue and the stock providers leave it at that -- measured on `tcp`, and read in the source for `efa`, whose CQ is a `util_cq` too. This provider guards them anyway, because it exists so that someone can find a bug in the code above it, and a consumer hunting its own threading mistake must not first have to rule out a corrupted ring inside the tool it is looking through |

The cost of that guard is worth stating plainly: a consumer that asks for `FI_THREAD_COMPLETION` and then forgets
the serialisation it owes works correctly here and breaks on the real device, and because the guard is a real lock
a sanitizer will not see the mistake either. The provider logs a warning naming the model when it guards a domain,
and `SRDSIM_FI_GUARD_CQ=0` leaves libfabric's choice alone. What is tested is the completion ring, under both models
libfabric disarms it for: `tests/fi/test_fi_cq_threads.cc` reads one queue from two threads, with
`FI_THREAD_COMPLETION` and again with `FI_THREAD_DOMAIN`, and checks that every completion comes back exactly
once. CI runs it a third time with the guard off to check that it then fails with a race reported inside that
ring. Binding or closing while another thread reads has no test of its own.

The endpoint options are answered as in libfabric's `prov/efa/src/rdm/efa_rdm_ep_fiops.c`. `FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES` is refused, as by the real RDM endpoint. `FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES` is accepted, and refused with `SRDSIM_FI_WRITE_IN_ORDER=0` ([configuration](configuration.md#libfabric-provider)), so a consumer's path for a device without it can be tested. `FI_OPT_EFA_EMULATED_READ` and `FI_OPT_EFA_EMULATED_WRITE` read false, and true once `FI_OPT_EFA_USE_DEVICE_RDMA` is set false; `FI_OPT_EFA_EMULATED_ATOMICS` reads true. `FI_HMEM` is offered whether or not a GPU is present, until device memory is carried.

## What it costs to run

The provider carries a consumer's traffic correctly; it does not carry it at the device's speed, and the gap is mostly one setting. Measured on a p5en.48xlarge (H200, one real EFA device present), NCCL over aws-ofi-nccl v1.21.1 with `OFI_NCCL_PROTOCOL=RDMA`, two ranks on two GPUs of the one machine told they are on different hosts so that NCCL uses the network plugin, one 8 MiB `all_reduce`, every row validated with no wrong values:

| Through | Time | Bandwidth |
|---|---|---|
| The real EFA device | 0.83 ms | 10.1 GB/s |
| This provider, defaults (8 KiB chunks, 200 us jitter) | 4.1 s | 0.002 GB/s |
| This provider, `SRDSIM_FI_JITTER_US=0`, 8 KiB chunks | 5.3 s | 0.002 GB/s |
| This provider, `SRDSIM_FI_JITTER_US=0` and `SRDSIM_CHUNK_BYTES=1048576` | 6.5 ms | 1.30 GB/s |
| The same, with `SRDSIM_MODE=inorder` | 6.5 ms | 1.29 GB/s |

The cost is per chunk, not the simulated jitter: removing the jitter at the default chunk size made it slower, not faster, while a 1 MiB chunk took the same transfer from 4.1 s to 6.5 ms. So for a test that moves data rather than one that checks orderings, raise `SRDSIM_CHUNK_BYTES`; a large chunk leaves fewer pieces to reorder, which is the thing the default size exists to produce.

## Testing

`tests/fi/test_fi_provider.cc` reaches the provider through libfabric, with the hints aws-ofi-nccl and NIXL build, and expects the answers libfabric's efa provider gives for its RDM endpoint. CI runs it against a libfabric built with its own efa provider, checks that the provider is loaded only when it is named, and checks that the test fails against a copy of the provider that claims write-after-write ordering.

CI also builds libfabric, the provider and these tests under AddressSanitizer and under ThreadSanitizer and runs the whole set again. That second pass is where the provider's agent thread is checked against the consumer's, which the plain pass cannot see at all. Two notes for anyone reproducing it by hand: libfabric has to be configured `--disable-opx`, because opx's `target_clones` atomics are the only ifuncs in `libfabric.so` and the loader resolves them before the executable's ThreadSanitizer runtime exists, so an instrumented resolver faults inside `ld.so` before `main`; and ThreadSanitizer has to be clang's, because gcc's warns that it cannot model the `atomic_thread_fence` this provider uses to publish a payload.

```bash
export PKG_CONFIG_PATH="$HOME/srdsim-libfabric/lib/pkgconfig" LD_LIBRARY_PATH="$HOME/srdsim-libfabric/lib"
cmake -S . -B build -DSRDSIM_BUILD_TESTS=OFF -DSRDSIM_BUILD_FI_TEST=ON \
  -DSRDSIM_FI_PROVIDER="$HOME/srdsim-libfabric/lib/srdsim/libsrdsim-fi.so"
cmake --build build -j --target test_fi_provider
ctest --test-dir build -R fi_provider --output-on-failure
```

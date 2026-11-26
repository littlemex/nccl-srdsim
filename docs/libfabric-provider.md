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
| NCCL over aws-ofi-nccl on one GPU host, with its send/recv and RDMA protocols | Runs: two ranks on one GPU, `all_reduce` correct under `srd` and `inorder` |
| NIXL's libfabric backend on one host | Runs: NIXL's own libfabric unit tests pass against this provider (backend, source-address resolution, rail refcount); its topology discovery reads the emulated NICs' PCIe addresses and link speed. Two-agent data transfer not yet run |

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

## What it reports

| Attribute | Value | Why |
|---|---|---|
| Endpoint | `FI_EP_RDM`, protocol `FI_PROTO_EFA`, address format `FI_ADDR_EFA`, 32-byte addresses | As the real RDM endpoint |
| Ordering | `FI_ORDER_SAS` for sends and receives, nothing for RMA | The real provider reorders two-sided messages back in software; SRD itself orders nothing, and hints that ask for an RMA ordering are refused |
| Registration | `FI_MR_VIRT_ADDR`, `FI_MR_ALLOCATED` and `FI_MR_PROV_KEY` for RMA, `FI_MR_HMEM` with `FI_HMEM`, and `FI_MR_LOCAL` when the hints offer it; 4-byte keys | As the real RDM endpoint, which uses a consumer's registrations when it offers them |
| Capabilities | Messages, tagged messages and RMA, with `FI_HMEM` only when asked for; no atomics | The real provider emulates atomics in software, and nothing this provider serves uses them |
| CQ data | 4 bytes | What `fi_writedata` carries on EFA |
| Progress | Manual, for data and control | As the real provider; hints asking for automatic progress are refused |

The endpoint options are answered as in libfabric's `prov/efa/src/rdm/efa_rdm_ep_fiops.c`. `FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES` is refused, as by the real RDM endpoint. `FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES` is accepted, and refused with `SRDSIM_FI_WRITE_IN_ORDER=0` ([configuration](configuration.md#libfabric-provider)), so a consumer's path for a device without it can be tested. `FI_OPT_EFA_EMULATED_READ` and `FI_OPT_EFA_EMULATED_WRITE` read false, and true once `FI_OPT_EFA_USE_DEVICE_RDMA` is set false; `FI_OPT_EFA_EMULATED_ATOMICS` reads true. `FI_HMEM` is offered whether or not a GPU is present, until device memory is carried.

## Testing

`tests/fi/test_fi_provider.cc` reaches the provider through libfabric, with the hints aws-ofi-nccl and NIXL build, and expects the answers libfabric's efa provider gives for its RDM endpoint. CI runs it against a libfabric built with its own efa provider, checks that the provider is loaded only when it is named, and checks that the test fails against a copy of the provider that claims write-after-write ordering.

```bash
export PKG_CONFIG_PATH="$HOME/srdsim-libfabric/lib/pkgconfig" LD_LIBRARY_PATH="$HOME/srdsim-libfabric/lib"
cmake -S . -B build -DSRDSIM_BUILD_TESTS=OFF -DSRDSIM_BUILD_FI_TEST=ON \
  -DSRDSIM_FI_PROVIDER="$HOME/srdsim-libfabric/lib/srdsim/libsrdsim-fi.so"
cmake --build build -j --target test_fi_provider
ctest --test-dir build -R fi_provider --output-on-failure
```

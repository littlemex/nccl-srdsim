# The libfabric provider

**English** | [日本語](libfabric-provider.ja.md)

`libsrdsim-fi.so` is a libfabric provider that registers under the name `efa`. Code that reaches EFA through libfabric, such as aws-ofi-nccl and NIXL's libfabric backend, takes its EFA branches only when the provider is called `efa`; this provider lets that code run between processes on one host, on an EFA instance without using its devices or on a host with no EFA at all, with the delivery rules of SRD. The design and its open questions are in [issue #75](https://github.com/littlemex/nccl-srdsim/issues/75).

## Status

| Part | State |
|---|---|
| Registration as `efa`, selection, `fi_getinfo` | Done |
| Fabric, domain, address vector, completion queue and endpoint objects; the EFA endpoint options | Done |
| Send, receive and RMA write between processes, over the fabric model the RMA plugin uses | Not yet: every transfer call returns `-FI_ENOSYS` |
| Write with immediate data, read, memory registration (host and CUDA) | Not yet |
| aws-ofi-nccl with nccl-tests, NIXL's libfabric transfer tests, on one host | Not yet |

## Building

The provider needs libfabric's headers and library, found through `pkg-config`. CI builds it against the libfabric release pinned in `versions.env`. On an EFA instance, build it against the libfabric the consumer uses, usually the one under `/opt/amazon/efa`.

```bash
export PKG_CONFIG_PATH=/opt/amazon/efa/lib/pkgconfig
cmake -S . -B build -DSRDSIM_BUILD_FI_PROVIDER=ON
cmake --build build -j --target srdsim-fi
```

## Selecting it

The provider is used only by a process that names it, in the preferred form of `FI_PROVIDER_PATH`: a `+` followed by the library's path. libfabric then gives it priority over any other provider called `efa`, including the real one, so on an EFA instance the process sees only the emulated device. It is never installed in libfabric's default provider directory, and a process that does not name it gets the real provider.

```bash
export FI_PROVIDER_PATH="+$PWD/build/libsrdsim-fi.so"
fi_info -p efa
```

`fi_info` then lists one domain, `srdsim0-rdm`.

## What it reports

| Attribute | Value | Why |
|---|---|---|
| Endpoint | `FI_EP_RDM`, protocol `FI_PROTO_EFA`, address format `FI_ADDR_EFA`, 32-byte addresses | As the real RDM endpoint |
| Ordering | `FI_ORDER_SAS` for sends and receives, nothing for RMA | The real provider reorders two-sided messages back in software; SRD itself orders nothing, and hints that ask for an RMA ordering are refused |
| Registration | `FI_MR_VIRT_ADDR`, `FI_MR_ALLOCATED`, `FI_MR_PROV_KEY`, and `FI_MR_HMEM` with `FI_HMEM`; not `FI_MR_LOCAL` | As the real RDM endpoint |
| CQ data | 4 bytes | What `fi_writedata` carries on EFA |
| Progress | Manual, for data and control | As the real provider; hints asking for automatic progress are refused |

The endpoint options are answered as in libfabric's `prov/efa/src/rdm/efa_rdm_ep_fiops.c`. `FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES` is refused, as by the real RDM endpoint. `FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES` is accepted, and refused with `SRDSIM_FI_WRITE_IN_ORDER=0` ([configuration](configuration.md#libfabric-provider)), so a consumer's path for a device without it can be tested. `FI_OPT_EFA_EMULATED_READ` and `FI_OPT_EFA_EMULATED_WRITE` read false.

## Testing

`tests/fi/test_fi_provider.cc` reaches the provider through libfabric, with the hints aws-ofi-nccl and NIXL build. CI runs it against a libfabric built with its own efa provider, checks that `fi_info` lists the provider only when it is named, and checks that the test fails against a copy of the provider that claims write-after-write ordering.

```bash
ctest --test-dir build -R fi_provider --output-on-failure
```

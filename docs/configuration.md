# Configuration

**English** | [日本語](configuration.ja.md)

Everything about a run that can be changed without rebuilding the plugin is an `SRDSIM_*` environment variable: which delivery order the simulator produces, how it copies into GPU memory, what it reports, and the switches that deliberately corrupt a run to prove a test would notice. Each group acts at one place on an operation's path from NCCL's call to the target's memory.

![The path of an operation from NCCL's call through the scheduler, the send side and the transport to the receive side. Above it, the settings that act at each stage: the schedule settings at the scheduler, staging and queueing at the send side, the address at the transport, the delivery context at the receive side. Below it, the settings that change the data at submit and on device writes, the diagnostics across the whole path, and the impairments, read at connect.](images/settings-map.svg)

The plugin reads each setting at one of four moments, so a value set after that moment has no effect. Set them all before NCCL initialises.

![A timeline of one process. The negative controls are read when the library is loaded; delivery and diagnostic settings the first time each is needed, and kept; the six schedule settings each time NCCL creates an RMA context; the published address each time it is published.](images/read-times.svg)

`scripts/check_configuration_reference.sh` fails CI when the variables the source reads and the rows of this document, or of its Japanese translation, disagree. It compares names only, not defaults or effects.

## How values are read

| Type | Accepted | Anything else |
|---|---|---|
| count | A decimal integer greater than 0 | Empty, `0`, a sign or a leading space (`-1`, `+8`), a value too large for 64 bits, or anything that is not a whole decimal number (`8k`, `0x2000`, `true`, a trailing space) gives the default |
| flag | A count: on for any positive integer | As for a count, so `0` and the empty string mean off, and so do `true` and `yes` |
| name | One of the names in the row | Any other value gives the default; the plugin logs it where the row says so |
| address | An IPv4 address, published as given | Not validated |

The plugin's log lines go to NCCL's logger, which shows them only with `NCCL_DEBUG=INFO`, and also to stderr when NCCL has not given the plugin a logger or `SRDSIM_LOG_OPS` is set.

## Schedule

Read each time NCCL creates an RMA context.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `SRDSIM_MODE` | name | `srd` | The delivery schedule: `srd`, `inorder`, `peer-seq`, `reverse-ops`, `signal-last`, `defer-one` or `violate-i1` (see the [README](../README.md#delivery-modes)). An unknown name is logged and `srd` is used |
| `SRDSIM_SEED` | count | `1` | Seed of the schedule's random choices. `0` gives the default, so seed 0 cannot be chosen |
| `SRDSIM_CHUNK_BYTES` | count | `8192` | Size of the chunks an operation is split into, and also, read once, the size of every pinned staging slot, which bounds `SRDSIM_STAGING_RING`. Use the same value on every rank: a frame larger than the receiver's slot is copied through pageable memory, and above 4 MiB a receiver accepts frames up to its own value only: a larger frame from a rank with a larger value closes the link, and the operations still to travel on it are lost. At most 64 MiB; a larger value is reduced to that |
| `SRDSIM_DEFER_KIND` | name | `any` | Which operation `defer-one` holds back: the oldest of `any` kind, or the oldest `unsignaled` put, `signaled` put or `signal-only` operation. An unknown name is logged and `any` is used |

## Delivery

Read once, the first time the plugin needs each, except `SRDSIM_LOCAL_IP`, which is read each time the plugin publishes its address. All but `SRDSIM_PROGRESS_ONE_CONTEXT` and `SRDSIM_LOCAL_IP` exist only in a build with `SRDSIM_WITH_CUDA`.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `SRDSIM_DELIVERY_CONTEXT` | name | the plugin's own CUDA context | Where copies into device memory are issued. The default and `green` need device windows allocated with `ncclMemAlloc` (VMM). `primary` uses the device's primary context, which accepts any allocation but makes each copy wait behind the application's module loads and frees on that context, the deadlock the default avoids. `green` uses a green context over the smallest SM partition, so a copy does not wait for a time slice while the application's kernels are resident; it shares the primary context's lock, and in a build with CUDA older than 12.4 registering a device window fails. Any other value gives the default |
| `SRDSIM_NO_GDRCOPY` | flag | off | Only in a build with `SRDSIM_WITH_GDRCOPY`. Pins and maps no window and delivers with CUDA copies instead, which do not complete while a resident kernel occupies the device |
| `SRDSIM_ASYNC_DELIVERY` | name | `1` | The exact value `0` waits for every CUDA copy as it is issued instead of queueing it; any other value gives the default. The queue is used only with CUDA copies (not GDRCopy) and pinned staging |
| `SRDSIM_STAGING_RING` | count | `128` | Pinned staging slots a thread may have outstanding when copies are queued. At least 2; capped so that the pinned memory stays under the plugin's limit, which depends on `SRDSIM_CHUNK_BYTES` |
| `SRDSIM_PAGEABLE_STAGING` | flag | off | Stages through pageable instead of pinned host memory, and waits for every copy. A reproduction mode |
| `SRDSIM_SIGNAL_KERNEL` | flag | off | Applies signals to device memory with a kernel launch instead of a store of the running total the plugin keeps |
| `SRDSIM_PROGRESS_ONE_CONTEXT` | flag | off | Advances only the context NCCL passes to `rmaProgress`, instead of every context of the communicator |
| `SRDSIM_LOCAL_IP` | address | the address of the default route | The address this rank publishes for its peers' connections |

## Diagnostics

These change what is reported, or check something at a cost; they leave the data delivered unchanged. `SRDSIM_TRACE` and `SRDSIM_DETECT_OVERLAP` are read with the schedule settings, and `SRDSIM_DETECT_OVERLAP` once more, once, for the check across contexts; the others are read once.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `SRDSIM_LOG_OPS` | flag | off | Logs every put, with its signal if it has one, and every get, with their arguments; copies every log line to stderr; starts the stall monitor; and installs the crash handler |
| `SRDSIM_TRACE` | flag | off | Logs every unit the schedule releases, to stderr |
| `SRDSIM_DETECT_OVERLAP` | count | off | `1` counts submissions that overlap an earlier unreleased write to the same peer within a context. `2` or more also checks across contexts: every submit then walks the operations in flight, and an overlap with a context whose lock is busy is skipped, so the check can miss an overlap but never reports one that is not there |
| `SRDSIM_PROBE_RECORD_BYTES` | count | off | Treats payloads that are a whole number of records of this size as records with a 4-byte non-negative stamp, and reports operations in which some records carry the stamp and others do not. Checked where the sender writes records to the wire, at submit when `SRDSIM_SNAPSHOT_SOURCE` is on, and, in a build with CUDA, where the receiver writes them to a device window |
| `SRDSIM_PROBE_STAMP_OFFSET` | count | `0` | Offset of the stamp within a record. When the stamp does not fit in the record the probe is off, without a message |
| `SRDSIM_STALL_MS` | count | `10000` | With `SRDSIM_LOG_OPS` set, which starts the stall monitor: milliseconds without an applied placement or signal before the monitor, which polls every 500 ms, reports where each thread last was. `0` gives the default |
| `SRDSIM_STALL_REPEAT_MS` | count | `5000` | Interval between repeated reports while a stall lasts |
| `SRDSIM_BACKTRACE` | flag | off | Installs the crash handler, which prints a backtrace on a fatal signal and then passes the signal on. Also installed by `SRDSIM_LOG_OPS` |
| `SRDSIM_NO_BACKTRACE` | flag | off | Never installs the crash handler; overrides `SRDSIM_BACKTRACE` and `SRDSIM_LOG_OPS` |

## Settings that change the data

Each can change the bytes a run delivers, on purpose. Read once; the three negative controls (`SRDSIM_CORRUPT_EVERY`, `SRDSIM_DROP_EVERY` and `SRDSIM_GET_LANDED_EARLY`) are read when the library is loaded.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `SRDSIM_DRY_RUN` | flag | off | Completes every operation at once without moving any data. Shows the sequence of calls; results are wrong |
| `SRDSIM_SNAPSHOT_SOURCE` | flag | off | Copies the whole device source of each put when it is submitted, sends that copy, and counts every unit whose source changed before release. Does nothing in a build without CUDA. Needs GDRCopy or synchronous delivery (`SRDSIM_ASYNC_DELIVERY=0` or `SRDSIM_PAGEABLE_STAGING`); otherwise it is ignored, with one line on stderr. Slow |
| `SRDSIM_CORRUPT_EVERY` | count | off | Negative control: flips one byte in the middle of every Nth device placement this rank applies. Does nothing on host windows or in a build without CUDA. Announced on stderr when active |
| `SRDSIM_DROP_EVERY` | count | off | Negative control: reports every Nth device placement as applied without writing it. Same scope and announcement |
| `SRDSIM_GET_LANDED_EARLY` | flag | off | Negative control: counts a get as landed when its last frame's copy into device memory has been queued, not when it has completed, so a flush can complete before the data is there. Does nothing in a build without CUDA, on host windows, with GDRCopy, or with copies waited for. Announced on stderr when active |

## Impairments

Each makes the fabric slow, bursty or partly stalled on purpose, never lossy, to expose liveness failures. Each communicator reads them when it connects and logs the ones in effect on stderr. The parser takes a decimal integer, with an empty or unparsable value giving the default; unlike the settings above, `0` means `0`.

| Variable | Type | Default | Sets |
|---|---|---|---|
| `SRDSIM_DELAY_US_MIN`, `SRDSIM_DELAY_US_MAX` | count | `0` | A random delay before each released unit; a maximum below the minimum is raised to it |
| `SRDSIM_BURST`, `SRDSIM_BURST_PAUSE_US` | count | `0` | Release in bursts of this many units, then a pause |
| `SRDSIM_APPLY_INTERVAL_US` | count | `0` | A wait before each frame the receiving side applies |
| `SRDSIM_SEND_BUFFER_BYTES` | count | `0` | The socket send buffer size (`SO_SNDBUF`) of each link the rank dials |
| `SRDSIM_FREEZE_PEER` | count | none | The peer whose link is frozen; `0` is a peer |
| `SRDSIM_FREEZE_AFTER_UNITS`, `SRDSIM_FREEZE_MS` | count | `0` | After this many units released to the frozen peer, release nothing more to it for this long (on the sending side, through the same room check as backpressure) |
| `SRDSIM_SIGNAL_LAG_US` | count | `0` | An extra delay before each signal |

## libfabric provider

Read by the libfabric provider (`libsrdsim-fi.so`, see [the provider's documentation](libfabric-provider.md)): `SRDSIM_FI_WRITE_IN_ORDER` each time the option it governs is set, the others when an endpoint is opened. The provider also reads `SRDSIM_MODE` (`srd` or `inorder` only), `SRDSIM_SEED` and `SRDSIM_CHUNK_BYTES`, with the meanings above.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `SRDSIM_FI_WRITE_IN_ORDER` | `0` or `1` | `1` | `1` accepts `FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES`, as an EFA device that supports it does; `0` refuses it with `-FI_EOPNOTSUPP`, as one that does not. Any other value is logged and `1` is used |
| `SRDSIM_FI_JITTER_US` | count | `200` | Under `srd`, the longest time, in microseconds, a write chunk waits at the target before it is applied; each waits a random time up to this. `0` applies chunks as they arrive. Not used under `inorder` |
| `SRDSIM_FI_CQ_DATA_EARLY` | `0` or `1` | `0` | `1` raises a write's remote completion with CQ data on its first chunk instead of its last, which breaks EFA's guarantee on purpose: the negative control for code that relies on it |

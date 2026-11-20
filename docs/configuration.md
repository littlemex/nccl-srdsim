# Configuration

The plugin reads its settings from `SRDSIM_*` environment variables. Set them before NCCL initialises: when each is read is listed below, and none is read again on the data path. `scripts/check_configuration_reference.sh` fails CI when the variables the source reads and the variables listed here disagree; it compares names only, not defaults or effects.

## How values are read

| Type | Accepted | Anything else |
|---|---|---|
| count | A decimal integer greater than 0 | Empty, `0`, or anything that is not a whole decimal number (`8k`, `0x2000`, `true`, a trailing space) gives the default. A leading `-` is accepted by the parser and wraps to a very large number, so do not use negative values |
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
| `SRDSIM_CHUNK_BYTES` | count | `8192` | Size of the chunks an operation is split into, and also, read once, the size of every pinned staging slot, which bounds `SRDSIM_STAGING_RING`. Use the same value on every rank: a frame larger than the receiver's slot is copied through pageable memory |
| `SRDSIM_DEFER_KIND` | name | `any` | Which operation `defer-one` holds back: the oldest of `any` kind, or the oldest `unsignaled` put, `signaled` put or `signal-only` operation. An unknown name is logged and `any` is used |

## Delivery

Read once, the first time the plugin needs each, except `SRDSIM_LOCAL_IP`, which is read each time the plugin publishes its address. All but `SRDSIM_PROGRESS_ONE_CONTEXT` and `SRDSIM_LOCAL_IP` exist only in a build with `SRDSIM_WITH_CUDA`.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `SRDSIM_DELIVERY_CONTEXT` | name | the plugin's own CUDA context | Where copies into device memory are issued. The default and `green` need windows allocated with `ncclMemAlloc` (VMM). `primary` uses the device's primary context, which accepts any allocation but makes each copy wait behind the application's module loads on that context, the deadlock the default avoids. `green` uses a green context over the smallest SM partition; it shares the primary context's lock, and in a build with CUDA older than 12.4 delivery fails. Any other value gives the default |
| `SRDSIM_NO_GDRCOPY` | flag | off | Only in a build with `SRDSIM_WITH_GDRCOPY`. Pins and maps no window and delivers with CUDA copies instead, which do not complete while a resident kernel occupies the device |
| `SRDSIM_ASYNC_DELIVERY` | name | `1` | The exact value `0` waits for every CUDA copy as it is issued instead of queueing it; any other value gives the default. The queue is used only with CUDA copies (not GDRCopy) and pinned staging |
| `SRDSIM_STAGING_RING` | count | `128` | Pinned staging slots a thread may have outstanding when copies are queued. At least 2; capped so that the pinned memory stays under the plugin's limit, which depends on `SRDSIM_CHUNK_BYTES` |
| `SRDSIM_PAGEABLE_STAGING` | flag | off | Stages through pageable instead of pinned host memory, and waits for every copy. A reproduction mode |
| `SRDSIM_SIGNAL_KERNEL` | flag | off | Applies signals to device memory with a kernel launch instead of a store of the running total the plugin keeps |
| `SRDSIM_PROGRESS_ONE_CONTEXT` | flag | off | Advances only the context NCCL passes to `rmaProgress`, instead of every context of the communicator |
| `SRDSIM_LOCAL_IP` | address | the address of the default route | The address this rank publishes for its peers' connections |

## Diagnostics

These change what is reported, or check something at a cost; they leave the data delivered unchanged. `SRDSIM_TRACE` and `SRDSIM_DETECT_OVERLAP` are read with the schedule settings; the others once.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `SRDSIM_TRACE` | flag | off | Logs every unit the schedule releases, to stderr |
| `SRDSIM_DETECT_OVERLAP` | count | off | `1` counts submissions that overlap an earlier unreleased write to the same peer within a context; `2` or more also checks across contexts. Every submit walks the operations in flight |
| `SRDSIM_PROBE_RECORD_BYTES` | count | off | Treats payloads that are a whole number of records of this size as records with a 4-byte non-negative stamp, and reports operations in which some records carry the stamp and others do not. The sender side is checked only with `SRDSIM_SNAPSHOT_SOURCE` |
| `SRDSIM_PROBE_STAMP_OFFSET` | count | `0` | Offset of the stamp within a record. When the stamp does not fit in the record the probe is off, without a message |
| `SRDSIM_STALL_MS` | count | `10000` | Milliseconds without an applied placement or signal before the stall monitor, which polls every 500 ms, reports where each thread last was. `0` gives the default, so the monitor cannot be turned off |
| `SRDSIM_STALL_REPEAT_MS` | count | `5000` | Interval between repeated reports while a stall lasts |
| `SRDSIM_BACKTRACE` | flag | off | Installs the crash handler, which prints a backtrace on a fatal signal and then passes the signal on. Also installed by `SRDSIM_LOG_OPS` |
| `SRDSIM_NO_BACKTRACE` | flag | off | Never installs the crash handler; overrides `SRDSIM_BACKTRACE` and `SRDSIM_LOG_OPS` |

## Settings that change the data

Each makes a run deliver something other than what was put, on purpose. Read once; the two negative controls are read when the library is loaded.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `SRDSIM_DRY_RUN` | flag | off | Completes every operation at once without moving any data. Shows the sequence of calls; results are wrong |
| `SRDSIM_SNAPSHOT_SOURCE` | flag | off | Copies the whole device source of each put when it is submitted, sends that copy, and counts every unit whose source changed before release. Needs a build with `SRDSIM_WITH_CUDA` and either GDRCopy or synchronous delivery (`SRDSIM_ASYNC_DELIVERY=0` or `SRDSIM_PAGEABLE_STAGING`); otherwise it is ignored with one log line. Slow |
| `SRDSIM_CORRUPT_EVERY` | count | off | Negative control: flips one byte in the middle of every Nth device placement this rank applies. Does nothing on host windows or in a build without CUDA. Announced on stderr when active |
| `SRDSIM_DROP_EVERY` | count | off | Negative control: reports every Nth device placement as applied without writing it. Same scope and announcement |

## Not applied

`impairments_from_env()` in `src/srdsim_impair.h` parses these, but the plugin does not call it, so setting them has no effect ([#18](https://github.com/littlemex/nccl-srdsim/issues/18)). The host tests set the same impairments directly. The parser takes a decimal integer, with an empty or unparsable value giving the default; unlike the settings above, `0` means `0`.

| Variable | Type | Default | Would set |
|---|---|---|---|
| `SRDSIM_DELAY_US_MIN`, `SRDSIM_DELAY_US_MAX` | count | `0` | A random delay before each released unit; a maximum below the minimum is raised to it |
| `SRDSIM_BURST`, `SRDSIM_BURST_PAUSE_US` | count | `0` | Release in bursts of this many units, then a pause |
| `SRDSIM_APPLY_INTERVAL_US` | count | `0` | At most one applied frame per interval on the receiving side |
| `SRDSIM_SEND_BUFFER_BYTES` | count | `0` | The socket send buffer size |
| `SRDSIM_FREEZE_PEER` | count | none | The peer whose link is frozen; `0` is a peer |
| `SRDSIM_FREEZE_AFTER_UNITS`, `SRDSIM_FREEZE_MS` | count | `0` | Freeze after this many applied units, for this long |
| `SRDSIM_SIGNAL_LAG_US` | count | `0` | An extra delay before each signal |

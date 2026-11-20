# Configuration

The plugin reads its settings from `SRDSIM_*` environment variables. The four schedule settings are read when NCCL creates each RMA context; every other setting is read once, the first time the plugin needs it, and kept for the life of the process, so set them before NCCL initialises. Every variable the source reads is listed here, and `scripts/check_configuration_reference.sh` fails CI when the two disagree.

Unless a row says otherwise, a number is a decimal integer, and an empty value, `0` or a value that does not parse means the default. A flag is on for any positive integer.

## Schedule

| Variable | Default | Effect |
|---|---|---|
| `SRDSIM_MODE` | `srd` | The delivery schedule: `srd`, `inorder`, `peer-seq`, `reverse-ops`, `signal-last`, `defer-one` or `violate-i1` (see the [README](../README.md#delivery-modes)). An unknown name is logged and `srd` is used |
| `SRDSIM_SEED` | `1` | Seed of the schedule's random choices. The same seed, mode and sequence of operations give the same schedule |
| `SRDSIM_CHUNK_BYTES` | `8192` | Size of the chunks an operation is split into before the schedule reorders them |
| `SRDSIM_DEFER_KIND` | `any` | Which operation `defer-one` holds back: the oldest of any kind, or the oldest `unsignaled` put, `signaled` put or `signal-only` operation. An unknown name is logged and `any` is used |

## Delivery

| Variable | Default | Effect |
|---|---|---|
| `SRDSIM_DELIVERY_CONTEXT` | the plugin's own CUDA context | `primary` copies into device memory from the device's primary context, which accepts any allocation; `green` copies from a green context (CUDA 12.4 or later). Any other value keeps the default, which needs windows allocated with `ncclMemAlloc` |
| `SRDSIM_NO_GDRCOPY` | off | In a build with GDRCopy, pins and maps no window and delivers with CUDA copies instead |
| `SRDSIM_ASYNC_DELIVERY` | `1` | `0` waits for every CUDA copy as it is issued instead of queueing it. Any other value keeps the default. The queue is used only with CUDA copies and pinned staging |
| `SRDSIM_STAGING_RING` | `128` | Number of pinned staging slots a thread may have outstanding when copies are queued; at least 2, and bounded by the plugin's limit on pinned memory |
| `SRDSIM_PAGEABLE_STAGING` | off | Stages through pageable instead of pinned host memory. Every copy is then waited for |
| `SRDSIM_SIGNAL_KERNEL` | off | Applies signals to device memory with a kernel launch instead of a store of the running total the plugin keeps |
| `SRDSIM_PROGRESS_ONE_CONTEXT` | off | Advances only the context NCCL passes to `rmaProgress`, instead of every context of the communicator |
| `SRDSIM_LOCAL_IP` | the address of the default route | The address this rank publishes for its peers' connections |

## Diagnostics

These change what is reported, or check something at a cost; none is needed for a correct run.

| Variable | Default | Effect |
|---|---|---|
| `SRDSIM_LOG_OPS` | off | Logs every put, with its signal if it has one, and every get, with their arguments; also installs the crash handler |
| `SRDSIM_TRACE` | off | Logs every unit the schedule releases |
| `SRDSIM_DRY_RUN` | off | Completes every operation at once without moving any data. Results are wrong by design; it shows the sequence of calls |
| `SRDSIM_DETECT_OVERLAP` | off | Counts submissions that overlap an earlier unreleased write to the same peer. Scans the pending operations on every submit |
| `SRDSIM_SNAPSHOT_SOURCE` | off | Copies the whole device source of each put when it is submitted, sends that copy, and counts every unit whose source changed before release. Synchronous delivery only (GDRCopy, or `SRDSIM_ASYNC_DELIVERY=0`); slow |
| `SRDSIM_PROBE_RECORD_BYTES` | off | Treats payloads as records of this many bytes and reports operations in which some records carry the stamp and others do not |
| `SRDSIM_PROBE_STAMP_OFFSET` | `0` | Offset of the stamp within a record. The probe is off when the stamp does not fit in the record |
| `SRDSIM_STALL_MS` | `10000` | Milliseconds without an applied placement or signal before the stall monitor reports where each thread last was |
| `SRDSIM_STALL_REPEAT_MS` | `5000` | Interval between repeated reports while a stall lasts |
| `SRDSIM_BACKTRACE` | off | Installs the crash handler, which prints a backtrace on a fatal signal and then passes the signal on. Also installed by `SRDSIM_LOG_OPS` |
| `SRDSIM_NO_BACKTRACE` | off | Never installs the crash handler, even with `SRDSIM_LOG_OPS` |

## Negative controls

Each makes a correct backend wrong on purpose, so that a test can show it would fail on a backend that corrupts or loses data. Counted per rank over device placements, and announced in the log when active.

| Variable | Default | Effect |
|---|---|---|
| `SRDSIM_CORRUPT_EVERY` | off | Flips one byte in the middle of every Nth placement |
| `SRDSIM_DROP_EVERY` | off | Reports every Nth placement as applied without writing it |

## Not applied

`impairments_from_env()` in `src/srdsim_impair.h` parses these, with `0` meaning off and no other default, but the plugin does not call it, so setting them has no effect ([#18](https://github.com/littlemex/nccl-srdsim/issues/18)). The host tests set the same impairments directly.

| Variable | Would set |
|---|---|
| `SRDSIM_DELAY_US_MIN`, `SRDSIM_DELAY_US_MAX` | A random delay before each released unit |
| `SRDSIM_BURST`, `SRDSIM_BURST_PAUSE_US` | Release in bursts of this many units, then a pause |
| `SRDSIM_APPLY_INTERVAL_US` | At most one applied frame per interval on the receiving side |
| `SRDSIM_SEND_BUFFER_BYTES` | The socket send buffer size |
| `SRDSIM_FREEZE_PEER`, `SRDSIM_FREEZE_AFTER_UNITS`, `SRDSIM_FREEZE_MS` | Stop accepting from one peer for a time, after a number of applied units |
| `SRDSIM_SIGNAL_LAG_US` | An extra delay before each signal |

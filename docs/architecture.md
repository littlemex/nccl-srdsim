# Architecture

**English** | [日本語](architecture.ja.md)

nccl-srdsim (the plugin, or the simulator: the same thing) is an NCCL plugin that delivers GPU-initiated writes in any order an unordered fabric such as SRD may produce. It lets code written for that fabric show, on an ordinary GPU host, the failures it would otherwise show only there.

## Background

GIN (GPU-Initiated Networking) is the NCCL API with which a GPU kernel writes into another rank's memory (a **put**), reads from it (a **get**), or increments a 64-bit counter there (a **signal**) that code on that rank waits on. A put can carry a signal, so that the waiting code learns the put's data is ready. In this document the **target** is the rank a put is written to and a **peer** is any other rank of the communicator.

NCCL hands these operations to an **RMA plugin** (remote memory access), a host library that carries them between ranks. SRD (Scalable Reliable Datagram, the transport of AWS EFA) is **reliable** but **unordered**: every message arrives exactly once, in any order. An RMA plugin for SRD therefore promises only what GIN's contract requires.

## The problem it solves

For an ordinary (weak) signal, GIN promises one ordering: the signal becomes visible only after every byte of its own put has been applied ([NCCL's RMA plugin specification](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/plugins/rma/README.md#L92-L97); this document calls it invariant **I1**). It promises nothing about a put issued earlier, even to the same target. Code that waits for a signal and then reads an earlier put is therefore wrong, but a transport that happens to deliver in order never exposes it.

![Rank 0 puts A without a signal, then B with signal S. On an in-order transport A, B and S arrive in that order, and a reader that waits for S and reads A gets the right value. On SRD, or this simulator, S may arrive before A, and the same reader reads a wrong A. Only B is ordered before S.](images/why.svg)

The simulator produces those legal reorderings on purpose. It can also break I1 on purpose (the `violate-i1` mode), so that you can confirm a test fails when the data a signal guards is incomplete.

## Terms

| Term | Meaning here |
|---|---|
| Operation | One call from NCCL: a put, a signalled put, a bare signal (a signalled put of size 0) or a get; and, on the owner of the memory, the reply to a get |
| Unit | What the scheduler releases: a placement (one chunk of a put's payload), a signal, or a get request |
| Release | The scheduler handing a unit to the send side; the order of release is what the delivery mode decides |
| Apply | The target's receiving agent writing a unit into the target's memory |
| Locally complete | What `test` reports: the operation's source may be reused. It says nothing about the target |
| Window | A registered memory region. Units name a window by number; each rank resolves the number to its own address |
| RMA context | NCCL's unit of ordering on a communicator. Each has its own scheduler; nothing is ordered across RMA contexts |
| Stream | The TCP connection carrying one direction between two ranks |

## Where the plugin sits

NCCL's GIN proxy backend (`NCCL_GIN_TYPE=2`) runs a host thread that takes each operation a GPU kernel posts and calls the RMA plugin's functions: `iput`, `iputSignal`, `iget`, `iflush`, `test` and `rmaProgress`. This plugin provides that table of functions (the op table). Its receiving half, not a NIC, performs every write into the target's memory, so the plugin chooses the order in which writes take effect.

![Rank A, sending: the GPU kernel's GIN calls go through NCCL's GIN proxy backend to this plugin's op table, whose scheduler and send queue release units onto one TCP stream per peer. Rank B, the target: the receiving agent writes each unit, in release order, into the data window or the signal window of its own memory.](images/architecture-overview.svg)

| Part | What it does |
|---|---|
| The application's GPU kernel | Issues puts, signals and gets through NCCL's GIN device API, and waits on signals |
| NCCL's GIN proxy backend | A host thread that turns each operation into a call to the plugin, and calls `rmaProgress` and `test` |
| This plugin, sending half | Splits each operation into units, chooses the order of release, and sends the units to the target |
| This plugin, receiving half | On the target, writes each unit into the target's own memory, in the order it was released |

Every rank runs the sending half and the receiving half: it sends its own operations and applies its peers'. What this means for code that will run on a different GIN backend in production is in [What is not simulated](#what-is-not-simulated).

## Life of a signalled put

A signalled put goes through six steps; the numbers match the figure.

![Six steps in a row: 1 submit (iputSignal returns at once), 2 split (8 KiB placements plus one signal unit), 3 release (rmaProgress pulls up to 8 units per RMA context), 4 send (in release order, on the target's stream), 5 apply (the target writes its own memory), 6 complete (test: the source may be reused; iflush: everything released). The sender's proxy thread runs steps 1 to 4 and 6; the target's receiving agent runs step 5.](images/put-lifecycle.svg)

1. **Submit.** NCCL's proxy thread calls `iputSignal` with the source, the size, the destination, the target rank and the signal. The plugin turns each memory handle into a window number, submits an operation to the RMA context's scheduler, and returns at once.
2. **Split.** The scheduler splits the payload into `ceil(size / chunk)` placement units (`SRDSIM_CHUNK_BYTES`, 8 KiB by default) and adds one signal unit.
3. **Release.** Each `rmaProgress` call asks the scheduler of every RMA context on the communicator for up to eight units. Which units may go, and in what order, is the delivery mode (next section). The randomised modes draw from a generator seeded with `SRDSIM_SEED`, so the same submissions and progress calls give the same order of release.
4. **Send.** Released units are written to the target's stream in release order. If a unit's source is device memory and copies are queued (CUDA copies with pinned staging, the default without GDRCopy), the unit is first copied to pinned host memory; the send queue sends no later unit to that target until the copy has finished, so the copy's timing cannot change the order the scheduler chose.
5. **Apply.** On the target, the receiving agent reads each frame, resolves the window number to its own address, checks the bounds, and writes the bytes or adds to the signal word.
6. **Complete.** `test` reports the operation done once all its units have been released and none is still in the send queue: the source may be reused, but nothing is known about the target. `iflush` completes when its RMA context's scheduler has released everything submitted to it, none of that context's units is still in the send queue, and every get that context issued has had its reply applied. It does not wait for puts to be applied at the target.

![A 32 KiB signalled put becomes placements P0 to P3 of 8 KiB and one signal S. Under srd the order P2, P0, P3, P1, S is legal because S comes after the last placement. Under violate-i1, S goes before every placement, which breaks I1. The target applies units in exactly the release order.](images/put-to-units.svg)

A TCP stream is reliable and ordered, so it adds no reordering of its own, and each receiving thread issues its writes in the order its frames arrive: a host write completes before the next frame is read, and device writes go onto one CUDA stream, or are waited for, in that order. The order of release on one RMA context is therefore the order in which that context's writes take effect at the target. What a seed does not fix is the interleaving between RMA contexts and between senders.

## The delivery contract and the modes

The scheduler follows one rule: reorder wherever the contract promises no order, and keep every order it does promise. Reordering less would hide the orderings the simulator exists to produce; reordering more would report failures that no conforming fabric can cause. The contract is:

- **I1.** A weak signal becomes visible only after every byte of its own put has been applied. Nothing is promised about earlier puts or signals.
- A **strong** signal also becomes visible only after every earlier put and signal of the same RMA context to the same target (see [Signals](#signals)).
- Puts are applied in any order relative to each other, also when they go to the same target, and the chunks of one put are applied in any order.

The scheduler chooses among the operations pending in an RMA context when NCCL calls `rmaProgress`. An operation that NCCL submits and the plugin releases before the next one is submitted cannot be overtaken, so the orders a run reaches depend on how NCCL calls the plugin as well as on the mode.

![Two operations to the same target: A with placements A0 and A1 and no signal, submitted first, and B with placements B0 to B2 and signal Bs. One row per mode shows an example order. srd: random per seed, and Bs before A0 is legal. inorder: submission order. peer-seq: A before B, chunks reordered. reverse-ops: newest operation first. signal-last: every payload, then the signals. defer-one: the oldest operation held to the end. violate-i1: a pending signal goes first, which breaks I1.](images/modes.svg)

`SRDSIM_MODE` selects the mode:

| Mode | Keeps I1 | What it is for |
|---|---|---|
| `srd` (default) | yes | A random legal order per seed |
| `inorder` | yes | Submission order within each RMA context. The control: code that fails here fails for a reason that has nothing to do with order |
| `peer-seq` | yes | Submission order per target, chunks within an operation reordered. More order than the contract promises, as aws-ofi-nccl's GIN implementation happens to keep. Code that passes here but fails under `srd` relies on more than the contract |
| `reverse-ops` | yes | Newest operation first, lowest chunk first: the simplest order that is not submission order |
| `signal-last` | yes | Every payload of every pending operation, then the signals: payload and signal as far apart as the contract allows |
| `defer-one` | yes | The oldest pending operation held back until every other has been released, as when one path is slow; chunk order is random |
| `violate-i1` | no | Releases a pending signal before anything else, whether or not its put has been applied, and ignores the strong-signal hold. A negative control |

## Threads, RMA contexts and streams

Each rank runs two kinds of thread for a communicator: NCCL's GIN proxy thread submits and progresses operations, and the plugin's receiving agent runs one thread per inbound stream that applies them.

![Rank 0, one communicator: NCCL's GIN proxy thread calls iput, iputSignal, iget, rmaProgress and test on two RMA contexts, each with its own scheduler and mutex; nothing is ordered across them. Both feed one send stream per peer. The receiving agent runs one thread per inbound stream and also submits the replies to gets. Ranks 1 and 2 have the same structure; every pair of ranks has a stream in each direction, and traffic to rank 0 itself goes through a socket pair along the same path.](images/threads.svg)

- **Streams.** In `listen`, each rank opens a TCP listening socket on all interfaces and returns, as the handle NCCL exchanges, the address its peers should dial: `SRDSIM_LOCAL_IP` if set, otherwise the address of the default route. In `connect`, each rank dials every peer and accepts a stream from every peer, so every pair of ranks has a stream in each direction. A rank's traffic to itself goes over a socket pair through the same path, because a NIC also delivers, and may reorder, writes a rank sends to itself. Sockets are IPv4, and both ends must be the same build on the same architecture, since frames are not converted between byte orders.
- **RMA contexts.** NCCL may create several on one communicator. Each has its own scheduler and mutex.
- **Threads.** NCCL's GIN proxy thread calls the submit functions, `rmaProgress` and `test`. On the target, the receiving agent (one per communicator) reads and applies each inbound stream on a thread of its own, and submits the reply when a frame is a get request.

## Components

The plugin is one source file and four headers it includes; a fifth header is used only by a test. The headers hold the logic whose correctness can be decided without a GPU, so they are tested on any Linux host.

![nccl_rma_srdsim.cc includes srdsim_schedule.h (order of release), srdsim_transport.h (frames and the receiving agent), srdsim_async.h (the send queue) and srdsim_mapping.h (BAR arithmetic), and uses NCCL's plugin headers and, optionally, the CUDA driver with NVRTC and GDRCopy. The headers need only the C++ standard library, and POSIX sockets for the transport. srdsim_impair.h is used only by the liveness test, not by the plugin.](images/components.svg)

| File | Responsibility |
|---|---|
| `src/srdsim_schedule.h` | The scheduler: splits each operation into units and decides the order of release according to the delivery mode |
| `src/srdsim_transport.h` | The transport: the frame format, the streams, the outbox per peer, and the receiving agent |
| `src/srdsim_async.h` | The send queue for units whose source is device memory still being copied to the host. A template over the unit type |
| `src/srdsim_mapping.h` | Address arithmetic for GDRCopy BAR mappings and the pinned staging pool |
| `src/srdsim_impair.h` | Deliberate impairments (delay, bursts, backpressure, a frozen link, a lagging signal), used by the liveness test; the plugin does not apply them yet ([#18](https://github.com/littlemex/nccl-srdsim/issues/18)) |
| `src/nccl_rma_srdsim.cc` | The op table, stream setup, window registration, delivery into device memory, completion and diagnostics |

## The transport and the receive side

The sending half never writes into the target's memory. It sends a frame header and the bytes over the stream, and the target's receiving agent writes into its own window. This needs neither GPU peer access between processes nor exporting an allocation across a process boundary, and it works the same within one host and across hosts.

![The receiving agent reads a frame header from the inbound stream, resolves the window number to its own address in the window table, checks that the window is known and the write is within bounds, then writes a host window with memcpy, a device window with a CUDA copy or GDRCopy, or adds to a signal word. A refused frame is skipped; if skipping would lose the start of the next frame, the stream is closed.](images/receive-side.svg)

A unit names a window by number, not by address. Window registration (`regMrSym`) is done by all ranks together, in the same order, so the numbers agree on every rank as long as every rank registers the same windows; each target resolves a number to its own base address.

Every frame is a 56-byte header followed by `length` bytes of payload. `length` always counts the bytes that follow, whatever the kind, which is what lets the receiving agent find the next frame after refusing one. A get request carries no payload, so its `length` is 0 and the number of bytes to read travels in `value`.

![The 56-byte header: kind, window, dest_window and requester (4 bytes each), then offset, length, value, op_id and dest_offset (8 bytes each), followed by length bytes of payload. Placement writes length bytes at window plus offset; Signal adds value to the 8 bytes at window plus offset; GetRequest reads value bytes (length is 0) and replies to requester at dest_window; GetResponse is a placement counted for the requester's RMA context named in op_id; Shutdown closes the stream. dest_window, requester and dest_offset are used by gets only.](images/wire-frame.svg)

| Layer | Responsible for | Retransmission and failure | Identifies |
|---|---|---|---|
| Stream (TCP) | Reliable, ordered bytes from one rank to another | TCP retransmits; a broken stream fails the RMA contexts sending on it | The pair of ranks and the direction |
| Outbox | Bytes released but not yet written to the stream, up to 64 MB per peer; `rmaProgress` pushes the rest | A frame that would exceed the limit is refused, and the RMA context that released it fails | The peer |
| Frame | One unit: its kind, where it goes, its length | Not retransmitted. A frame naming an unknown window or running past its window is skipped; one longer than the largest frame the transport emits, or of an unknown kind, closes the stream | `window` and `offset` locate the write; for gets, `op_id` names the requester's RMA context |
| Window table | Turning a window number into this rank's address | A number no longer registered is refused | `window` |

A skipped placement is lost, and the signal of its operation is still applied when it arrives; a lost get reply leaves `iflush` waiting. Both are error paths a conforming peer never takes, and both are tracked in [#24](https://github.com/littlemex/nccl-srdsim/issues/24).

## Delivery into device memory

The hazard on a GPU is a deadlock: the application's kernels stay resident on the GPU, busy-waiting for data, while the copy that delivers the data is queued behind something the application holds. The four ways of writing a device window differ in what a copy can be queued behind.

![Four columns. Own CUDA context (the default): a CUDA copy from a context the plugin made; needs VMM windows; does not wait for the application's module loads. Primary (SRDSIM_DELIVERY_CONTEXT): a CUDA copy from the primary context; accepts any allocation; waits behind module loads and frees and can deadlock. Green: a CUDA copy on the smallest SM partition; needs VMM windows and CUDA 12.4; no time-slice wait, but shares the primary context's lock. GDRCopy (build option): the CPU writes through the GPU's BAR mapping; needs /dev/gdrdrv and VMM windows; never waits for the GPU.](images/delivery-contexts.svg)

| Way | What it does | What it needs | What a write can wait behind |
|---|---|---|---|
| Own CUDA context (default) | Copies from a CUDA context the plugin creates for itself | Device windows allocated with `ncclMemAlloc` (VMM allocations, whose mappings belong to the process rather than to one CUDA context); registration refuses any other | Nothing the application does on its own context, such as loading a module inside a dispatch (DeepEP's JIT does) |
| Primary (`SRDSIM_DELIVERY_CONTEXT=primary`) | Copies from the device's primary CUDA context | Any allocation | The application's module loads and frees on that context: the deadlock above. It exists to reproduce it |
| Green (`SRDSIM_DELIVERY_CONTEXT=green`) | Copies from a green context, a CUDA context limited to the smallest group of SMs (streaming multiprocessors) the device allows | VMM windows and CUDA 12.4 or later | Not a time slice on the GPU, but the primary context's lock, which a module load holds; suits applications that load and free nothing while their kernels wait |
| GDRCopy (build option) | Maps the window into the process through the GPU's BAR (its PCIe memory aperture), and the CPU writes it | `/dev/gdrdrv`, and VMM windows unless the context is `primary`. A window not aligned to a GPU page falls back to a CUDA copy that is waited for | Nothing on the GPU |

## Signals

A signal unit adds to a 64-bit word, the signal word, in the target's signal window: the signal's value for `NCCL_GIN_SIGNAL_OP_ADD`, and one otherwise.

![Three panels. Host window: the receiving agent writes the payload, then adds to the signal word with memory_order_release, which pairs with the reader's memory_order_acquire load, after which the reader sees every payload byte; T12 checks this under ThreadSanitizer. Device window: by default the plugin stores the running total it keeps, which needs no SM; with SRDSIM_SIGNAL_KERNEL an atomic add runs in a kernel compiled with NVRTC; the store is valid because NCCL never resets these words. Strong signal: S! is held on its RMA context and target until A and B are released.](images/signals.svg)

- **Host windows.** The agent adds to the word atomically with `memory_order_release`, after the payload bytes it placed from the same stream. A reader that loads the word with `memory_order_acquire`, as a GIN receiver does, then sees those bytes. The transport test T12 is such a reader; with the add made relaxed, ThreadSanitizer reports it.
- **Device windows.** By default the plugin keeps the running total of each signal word, starting from zero, and stores it as eight aligned bytes. NCCL registers the signal region so that its words are never reset (it remembers offsets instead), which makes the plugin the only writer and a store of the total equivalent to an atomic add. No kernel is launched, so delivery needs no SM while the application's kernels occupy them all. `SRDSIM_SIGNAL_KERNEL=1` applies the add with a small kernel compiled at run time with NVRTC instead, to compare the two.
- **Strong signals.** SRD has no strong signal, but NCCL may request one, and a submit call that returns an error stops NCCL's GIN proxy thread for good. So the plugin accepts strong signals, and the scheduler holds each one until every earlier operation of that RMA context to that target has been released. In the figure, S! waits for the earlier operations A and B. With NCCL headers that define the v16 op table, the plugin also reports `flushesAllPutsOnAnySignal` as false: a signal does not imply that puts from other peers have been applied.

## Gets

A get is answered by the owner of the memory, which sends the data back the way it would send a put.

![1: iget submits a GetRequest unit, which is released like any unit. 2: the owner's receiving agent submits the reply on the owner's first RMA context. 3: the owner's rmaProgress releases the reply in chunks. 4: the GetResponse chunks travel back, reordered like any put. 5: the requester counts each chunk applied, and iflush waits for all of them. A get is locally complete once its request has been sent.](images/get-flow.svg)

Because the reply goes through the owner's scheduler, it is chunked and reordered like any put and leaves only when the owner's `rmaProgress` runs. EFA-GDA has no get; this path exists for code written against GIN backends that have one.

## Impairments and diagnostics

The plugin does not yet apply impairments ([#18](https://github.com/littlemex/nccl-srdsim/issues/18)): `src/srdsim_impair.h` models a slow, bursty, congested or partly frozen fabric, never a lossy one, and only the liveness test uses it. Such impairments target liveness failures, where something stops making progress, which are often the hardest defects to find on real hardware; ordering failures, where some reachable state is wrong, are what the delivery modes target.

For diagnosis, `SRDSIM_LOG_OPS=1` logs every operation and also starts a stall monitor, which reports where each thread last was when no unit has been applied for `SRDSIM_STALL_MS`. Every setting is listed in [the configuration reference](configuration.md).

## Testing

The properties that can be decided without a GPU are tested on every pull request, on the headers. The plugin itself runs under NCCL only in the GPU test, which is run by hand.

![A table of tests against what each exercises. The scheduler: test_srdsim_schedule, model_srdsim_schedule_exhaustive, test_srdsim_adversarial and test_srdsim_liveness. The transport: test_srdsim_transport, model_srdsim_framing, test_srdsim_adversarial, test_srdsim_get and test_srdsim_liveness. The send queue: test_srdsim_async. The mapping: test_srdsim_mapping. The plugin with NCCL: CI's plugin builds and exports, and tests/gpu run by hand on a GPU. Host tests also run under ASan, UBSan and TSan.](images/testing-map.svg)

| Test | What it establishes |
|---|---|
| `test_srdsim_schedule` | For each of the seven modes over 64 seeds: every submitted byte is released exactly once and to the right place; the six modes that keep I1 never release a signal early, and `violate-i1` does; each mode's defining order holds; a seed reproduces its order. Over 60,000 seeds, a case of two operations reaches all 2,520 legal orders and no illegal one |
| `model_srdsim_schedule_exhaustive` | Every state the scheduler can reach, for configurations small enough: in `srd` and `inorder` none releases a signal before its put, in `violate-i1` at least one does, every unit is released exactly once, and every path ends |
| `test_srdsim_transport` | Over a real TCP connection: every byte and signal arrives exactly once, the order of application equals the order of release, and a reader that acquire-loads a signal word sees the bytes placed before it (T12) |
| `model_srdsim_framing` | Every sequence of the frame kinds and outcomes it models: after each frame the receiving agent is at the start of the next one or the stream is closed |
| `test_srdsim_adversarial` | A hostile peer and misuse: an enormous frame length, an overflowing `offset + length`, an operation id never issued, a zero chunk size, a shutdown racing a sender. Each is refused or survived |
| `test_srdsim_async` | The send queue keeps per-target order whatever order the copies complete in, and an operation stays incomplete until its last unit has left |
| `test_srdsim_get` | Get requests and replies carry the right source, destination and owner |
| `test_srdsim_mapping` | The BAR mapping and staging-pool arithmetic, at every slot and across page offsets, including values chosen to overflow naive arithmetic |
| `test_srdsim_liveness` | Bounds on elapsed time under impairment: slow is told apart from never |

CI also runs the host tests under AddressSanitizer, UBSan and ThreadSanitizer; builds the plugin against the headers of each supported NCCL release in each delivery configuration, and with CUDA 12 and 13 with and without GDRCopy; checks that the op tables NCCL looks up are exported; and loads the host-only build with `dlopen`. The op table, stream setup, the delivery contexts, completion and gets are exercised end to end only by the GPU test ([#8](https://github.com/littlemex/nccl-srdsim/issues/8)). The strong-signal hold has no host test yet.

## What is not simulated

The simulator reproduces the order of delivery through NCCL's GIN proxy backend, and nothing below it.

![Reproduced: RMA through NCCL's GIN proxy backend; contract-legal orders among pending operations; orders that break it on purpose; puts, signals, strong signals and gets; host and device windows. Not reproduced: collectives, which use the network plugin interface; device-side GIN (GDAKI, EFA-GDA), which has no host on the path; NIC queues, doorbells and PCIe order, which are hardware; loss and duplication, since SRD is reliable; bandwidth and latency, since data goes through host memory.](images/scope.svg)

| Not simulated | Why |
|---|---|
| NCCL's collectives (all-reduce, all-gather, send and receive) | They use the network plugin interface, which this plugin does not implement |
| Device-side GIN backends (GDAKI, EFA-GDA) and anything built on NVSHMEM | The GPU drives the NIC directly; no host op table is on the path. Code that will use one in production is tested here against the contract that backend promises, not on that backend |
| What the fabric can advertise | NCCL's proxy backend reports strong and VA signals as supported for every RMA plugin, so the plugin cannot say that SRD lacks them; it implements strong signals instead |
| Work-request encoding, doorbells, completion-queue limits, PCIe and MMIO ordering | These are properties of the device path and the hardware memory model |
| Loss and duplication | SRD is reliable |
| Bandwidth, latency, anything measured in seconds | Payloads are staged through host memory; the simulator makes no performance claim |

# Architecture

**English** | [日本語](architecture.ja.md)

nccl-srdsim is an NCCL RMA plugin that delivers the puts, signals and gets of NCCL's GIN API in any order that an unordered, reliable fabric such as SRD is allowed to produce. It lets code written against GIN show, on an ordinary GPU host, the failures it would otherwise show only on that fabric.

## The problem it solves

GIN's contract for a weak signal promises one ordering: the signal becomes visible only after every byte of its own put has landed ([NCCL's RMA plugin specification](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/plugins/rma/README.md#L92-L97); this document calls it invariant I1). An earlier put to the same peer is not ordered before that signal. A test setup whose transport happens to deliver in order never shows the difference, so code that also relies on the earlier put passes there and fails on SRD.

![Rank 0 puts A without a signal, then B with signal S. On an in-order transport A, B and S arrive in that order, and a reader that waits for S and reads A gets the right value. On SRD, or this simulator, S may arrive before A, and the same reader reads a wrong A. Only B is ordered before S.](images/why.svg)

The simulator produces those legal reorderings deliberately, and reproducibly from a seed. It can also break I1 on purpose (the `violate-i1` mode), so that a test can show it really checks the data its signal guards.

## Where the plugin sits

NCCL's proxy GIN backend (`NCCL_GIN_TYPE=2`) has a host thread take each operation a GPU kernel posts and call an RMA plugin: `iput`, `iputSignal`, `iget`, `iflush`, `test` and `rmaProgress`. This plugin is that RMA plugin. Because it performs every write itself, it decides the order in which they land, and it needs no device code to do so.

![Rank A, sending: the GPU kernel's GIN calls go through NCCL's GIN proxy backend to this plugin's op table, whose scheduler and send queue release units onto one TCP stream per peer. Rank B, the target: the receiving agent writes each unit, in release order, into the data window or the signal window of its own memory.](images/architecture-overview.svg)

| Part | What it does |
|---|---|
| The application's GPU kernel | Issues puts, signals and gets through NCCL's GIN device API, and waits on signals |
| NCCL's GIN proxy backend | A host thread that turns each operation into a call to the plugin, and calls `rmaProgress` and `test` |
| This plugin, sending | Splits each operation into units, chooses the order in which they are released, and sends them to the peer |
| This plugin, receiving | On the target, writes each unit into the target's own memory, in the order it was released |

Every rank runs both halves. The plugin exports the `ncclRmaPlugin_v14` op table, and also `ncclRmaPlugin_v15` and `ncclRmaPlugin_v16` when the NCCL headers it is built against define them.

Two limits follow. Only traffic through the proxy backend is simulated: code that will use a device-side backend (GDAKI, EFA-GDA) in production is tested here against the contract that backend promises, not on that backend. And the plugin cannot advertise a fabric's capabilities, because NCCL's proxy backend reports strong and VA signals as supported on its behalf; the plugin imitates the delivery order instead, and implements strong signals rather than refusing them (see [Signals](#signals)).

## Terms

| Term | Meaning here |
|---|---|
| Operation | One call from NCCL: a put, a signalled put, a bare signal (a signalled put of size 0) or a get |
| Unit | What the scheduler releases: a placement (one chunk of an operation's payload), a signal, or a get request |
| Released | Handed by the scheduler to the send side |
| Applied | Written into the target's memory by the target's receiving agent |
| Locally complete | What `test` reports: the operation's source may be reused. It says nothing about the target |
| Window | A registered memory region. Units name a window by number; each rank resolves the number to its own address |

## Components

The plugin is one source file built on four headers. The headers hold the logic whose correctness can be decided without a GPU, and each includes only the C++ standard library, so they are tested on any host.

![nccl_rma_srdsim.cc includes srdsim_schedule.h (order of release), srdsim_transport.h (frames and the receiving agent), srdsim_async.h (the send queue) and srdsim_mapping.h (BAR arithmetic), and uses NCCL's plugin headers and, optionally, the CUDA driver with NVRTC and GDRCopy. srdsim_impair.h is used only by the liveness test, not by the plugin.](images/components.svg)

| File | Responsibility |
|---|---|
| `src/srdsim_schedule.h` | The scheduler: splits each operation into units and decides the order of release according to the delivery mode |
| `src/srdsim_transport.h` | The wire: the frame format, one TCP stream per peer on the sender, and a receiving agent per inbound stream on the target |
| `src/srdsim_async.h` | The send queue for units whose source is device memory still being copied to the host. A template over the unit type |
| `src/srdsim_mapping.h` | Address arithmetic for GDRCopy BAR mappings and the pinned staging pool |
| `src/srdsim_impair.h` | Deliberate impairments (delay, bursts, backpressure, a frozen link, a lagging signal). Used by the liveness test; the plugin does not apply them yet ([#18](https://github.com/littlemex/nccl-srdsim/issues/18)) |
| `src/nccl_rma_srdsim.cc` | The op table NCCL loads, connection setup, window registration, delivery into device memory, completion and diagnostics |

## Threads, contexts and connections

Each rank has one thread that submits and progresses operations, NCCL's, and one thread per inbound connection that applies them, the plugin's. Ordering is decided per RMA context.

![Rank 0: NCCL's GIN proxy thread calls submit, rmaProgress and test on two RMA contexts, each with its own scheduler and mutex; nothing is ordered across contexts. Both feed one send stream per peer. Receiving agents run one thread per inbound stream. Ranks 1 and 2 have the same structure; every pair of ranks has a stream in each direction, and traffic to rank 0 itself goes through a socket pair along the same path.](images/threads.svg)

- **Connections.** In `listen`, each rank opens a TCP listening socket (on `SRDSIM_LOCAL_IP`, or the address of the default route) and returns its address as the handle NCCL exchanges. In `connect`, each rank dials every peer and accepts a connection from every peer, so every pair has a stream in each direction. A rank's traffic to itself goes over a socket pair through the same path, because a NIC also delivers, and may reorder, writes a rank sends to itself.
- **Contexts.** NCCL may create several RMA contexts on one communicator. Each has its own scheduler and mutex. Ordering, including the strong-signal hold, is per context and peer.
- **Threads.** NCCL's GIN proxy thread calls the submit entry points, `rmaProgress` and `test`. On the target, one receiving-agent thread per inbound stream reads and applies that stream's frames.

## Life of a signalled put

A signalled put goes through six steps; the numbers match the figure.

![Six steps in a row: 1 submit (iputSignal returns at once), 2 split (8 KiB placements plus one signal unit), 3 release (rmaProgress pulls up to 8 units per context), 4 send (in release order, on the peer's stream), 5 apply (the target writes its own memory), 6 complete (test: the source may be reused; iflush: applied). The sender's proxy thread runs steps 1 to 4 and 6; the target's receiving agent runs step 5.](images/put-lifecycle.svg)

1. **Submit.** The proxy thread calls `iputSignal` with the source, the size, the destination, the target rank and the signal. The plugin translates each memory handle into a window number, submits an operation to the context's scheduler and returns at once. An operation larger than 1 GiB is refused.
2. **Split.** The scheduler splits the payload into `ceil(size / chunk)` placement units (`SRDSIM_CHUNK_BYTES`, 8 KiB by default) and adds one signal unit.
3. **Release.** Release is pulled, not pushed: each `rmaProgress` call asks the scheduler of every context on the communicator for up to eight units (`SRDSIM_PROGRESS_ONE_CONTEXT=1` advances only the calling context). Which units may go, and in what order, is the delivery mode. The randomised modes draw from a generator seeded with `SRDSIM_SEED`, so the same submissions and progress calls give the same order.
4. **Send.** Released units are written to the peer's stream in release order. When device memory is copied with CUDA (the default without GDRCopy), a unit whose source is device memory is first copied to a pinned host slot asynchronously, and the send queue holds every later unit to that peer until the copy has landed, so the copy's timing cannot change the order the scheduler chose.
5. **Apply.** On the target, the receiving agent reads each frame, resolves the window number to its own address, checks the bounds and writes the bytes, or adds to the signal word.
6. **Complete.** `test` reports the operation done when every unit has been released and none is still in the send queue: the source may be reused, as with a NIC's local completion, but nothing is known about the target. `iflush` completes when the scheduler of the context it was issued on is idle, none of that context's units is in the send queue, and every get that context waits for has been applied. Its `rank` argument does not narrow it to one peer.

![A 32 KiB signalled put becomes placements P0 to P3 of 8 KiB and one signal S. Under srd the order P2, P0, P3, P1, S is legal because S comes after the last placement; under violate-i1 the order P2, S, P0, P3, P1 breaks I1 because S is visible while P0, P1 and P3 are missing. The target applies units in exactly the release order.](images/put-to-units.svg)

A TCP stream is reliable and ordered, so it adds no reordering of its own. Every reordering comes from the scheduler, where a seed reproduces it, and a placement is fully written before the agent reads the next frame, so the order of release is the order in which writes become visible at the target.

## The delivery contract and the modes

The scheduler follows one rule: reorder wherever the contract promises no order, and keep every order it does promise. Reordering less would hide the orderings the simulator exists to produce; reordering more would report failures that no conforming fabric can cause. The contract is:

- **I1.** A signal becomes visible only after every byte of its own put has been applied. Nothing is promised about earlier puts or signals.
- Puts are applied in any order relative to each other, also when they go to the same peer.
- The chunks of one put are applied in any order.

![Two operations to the same peer, A with placements A0 to A2 and signal As submitted first, and B with B0 and B1. One row per mode shows its release order: srd one random legal order per seed; inorder submission order; peer-seq A before B with chunks reordered; reverse-ops newest operation first; signal-last every payload, then the signals; defer-one the oldest operation held back to the end; violate-i1 As before A1 and A2, which breaks I1. Every mode but violate-i1 keeps I1.](images/modes.svg)

`SRDSIM_MODE` selects the mode:

| Mode | Keeps I1 | What it is for |
|---|---|---|
| `srd` (default) | yes | Any legal order, drawn at random per seed |
| `inorder` | yes | Submission order. The control: code that fails here fails for a reason that has nothing to do with order |
| `peer-seq` | yes | Submission order per peer, chunks within an operation reordered. Stronger than the contract; it matches how aws-ofi-nccl's GIN implementation retires received operations ([`rx_consumed`](https://github.com/aws/aws-ofi-nccl/blob/2475c4c736907e68ebeadb3848f31231ee4a44d0/include/rdma/gin/nccl_ofi_gin_types.h#L86-L89)). Code that passes here but fails under `srd` relies on more than the contract |
| `reverse-ops` | yes | Newest operation first: the simplest order that is not submission order |
| `signal-last` | yes | Every payload of every pending operation, then the signals: payload and signal as far apart as the contract allows |
| `defer-one` | yes | The oldest pending operation held back until every other has been released, as when one path is slow. With two operations, as in the figure, it equals `reverse-ops` |
| `violate-i1` | no | May release a signal before the rest of its put. A negative control: code that depends on I1 should fail under it for some seed |

## The wire and the receive side

The sender never writes into the target's memory. It ships a descriptor and the bytes over the stream, and the target's receiving agent writes into its own window. This needs neither GPU peer access between processes nor exporting an allocation across a process boundary, and it works the same within one host and across hosts.

![The receiving agent reads a frame header from the peer's stream, resolves the window number to its own address in the window table, checks that the window is known and the write is within bounds, then writes a host window with memcpy, a device window with a CUDA copy or GDRCopy, or adds to a signal word. A refused frame is skipped; if skipping would lose the start of the next frame, the link is closed.](images/receive-side.svg)

A unit names a window by number, not by address, as a NIC's put names a remote key. Window registration (`regMrSym`) is collective, so every rank registers the same windows in the same order and the numbers agree everywhere; each target resolves a number to its own base address.

Every frame is a 56-byte header followed by `length` bytes of payload. The header's `length` always counts the bytes that follow, whatever the kind, which is what lets the receiver find the next frame after refusing one.

![The 56-byte header: kind, window, dest_window and requester (4 bytes each), then offset, length, value, op_id and dest_offset (8 bytes each), followed by length bytes of payload. Placement writes length bytes at window plus offset; Signal adds value to the 8 bytes at window plus offset; GetRequest reads value bytes and replies to requester at dest_window; GetResponse is a placement counted toward the get that op_id names; Shutdown closes the stream. dest_window, requester and dest_offset are used by gets only.](images/wire-frame.svg)

| Layer | Responsible for | Retransmission and failure | Identifies |
|---|---|---|---|
| TCP stream | Reliable, ordered bytes between one pair of ranks, one per direction | TCP retransmits; a broken stream ends the link | The pair of ranks |
| Frame | One unit: its kind, where it goes, its length | None. A frame naming an unknown window or running past its window is skipped; one whose length is impossible closes the link | `window` and `offset` locate the write; `op_id` names the operation |
| Window table | Turning a window number into this rank's address | A number no longer registered is refused | `window`, the same number on every rank |

## Delivery into device memory

When a window is device memory, the receiving agent writes it in one of four ways. They differ in what a write can end up waiting behind while the application's kernels sit on the GPU waiting for the very data the write delivers.

![Four columns. Own context (the default): a CUDA copy from a context the plugin made; needs VMM windows; does not wait behind the application's module loads. Primary (SRDSIM_DELIVERY_CONTEXT): a CUDA copy from the primary context; accepts any allocation; waits behind module loads and frees and can deadlock. Green: a CUDA copy on the smallest SM partition; needs VMM windows and CUDA 12.4; no time-slice wait, but shares the primary context's lock. GDRCopy (build option): the CPU writes through the GPU's BAR mapping; needs /dev/gdrdrv; never waits for the GPU.](images/delivery-contexts.svg)

- **Own context** (default). A copy issued from the device's primary context can wait behind a module load the application performs on that context, and some do so inside a dispatch (DeepEP's JIT, for example). A copy from a second context does not. The windows must then be VMM allocations (`ncclMemAlloc`), whose mappings belong to the process rather than to one context, and registration refuses any other kind of device window.
- **Primary** (`SRDSIM_DELIVERY_CONTEXT=primary`). Accepts any allocation; it exists to reproduce that deadlock.
- **Green** (`SRDSIM_DELIVERY_CONTEXT=green`). Copies over the smallest SM partition the device allows, so a copy does not wait for a time slice while kernels are resident. A green context shares the primary context's lock, so it suits only applications that load no modules and free nothing while their kernels wait.
- **GDRCopy.** With GDRCopy built in and `/dev/gdrdrv` present, device windows are mapped into the process through the GPU's BAR and written by the CPU, which does not wait for the GPU at all.

## Signals

A signal unit adds to a 64-bit word in the target's signal window: the signal's value for `NCCL_GIN_SIGNAL_OP_ADD`, and one otherwise.

![Three panels. Host window: the receiving agent writes the payload, then adds to the signal word with release order, which pairs with the reader's acquire load, after which the reader sees every payload byte; T12 checks this under ThreadSanitizer. Device window: by default the plugin stores the running total it keeps, which needs no SM; with SRDSIM_SIGNAL_KERNEL an atomic add runs in a kernel compiled with NVRTC; the store is valid because NCCL never resets these words. Strong signal: S! is held on its context and peer until A and B are released.](images/signals.svg)

- **Host windows.** The update is an atomic add with release order, made after the payload bytes the agent placed before it. A reader that loads the word with acquire order, as a GIN receiver does, then sees those bytes. A reader that forgets the acquire is still wrong, and ThreadSanitizer still reports it.
- **Device windows.** By default the plugin keeps the running total of each signal word and stores it as eight aligned bytes. NCCL registers the signal region so that its words are never reset (it remembers offsets instead), which makes the plugin the only writer and a store of the total equivalent to an atomic add. No kernel is launched, so delivery never needs an SM while the application's kernels occupy them all. `SRDSIM_SIGNAL_KERNEL=1` applies the add with a small kernel compiled at run time with NVRTC instead, to compare the two.
- **Strong signals.** A strong signal orders every earlier put and signal on the same context and peer before itself. SRD has no such operation, but NCCL may request one, and a submit call that returns an error stops NCCL's GIN progress thread for good. So the plugin accepts strong signals, and the scheduler holds one until every earlier operation on that context and peer has been released.

## Gets

A get is answered by the owner of the memory, which sends the data back the way it would send a put.

![The requester sends a GetRequest naming window, offset and length (1). The owner submits the reply on its own first RMA context (2), and its rmaProgress releases the reply in chunks (3). The GetResponse chunks travel back, reordered like any put (4). The requester counts each chunk as it lands, and iflush waits for all of them (5). A get is locally complete once its request has been sent.](images/get-flow.svg)

Because the reply goes through the owner's scheduler, it is chunked and reordered like any put and leaves only when the owner's `rmaProgress` runs. EFA-GDA has no get; this path exists for code written against the proxy backends that have one.

## Impairments and diagnostics

Ordering failures are safety failures: some reachable state is wrong. The defects hardest to find on real hardware are often liveness failures instead: something stops making progress. `src/srdsim_impair.h` models a slow, bursty, congested or partly frozen fabric, never a lossy one, since SRD is reliable. The liveness test drives the transport with these impairments; the plugin parses their environment variables but does not apply them yet ([#18](https://github.com/littlemex/nccl-srdsim/issues/18)).

For diagnosis, `SRDSIM_LOG_OPS=1` logs every operation, and a stall monitor reports where each thread last was when no unit has been applied for `SRDSIM_STALL_MS`. Every setting is listed in [the configuration reference](configuration.md).

## Testing

The properties that can be decided without a GPU are tested on every pull request, on the headers; the plugin itself is exercised under NCCL by the GPU test.

![A table of tests against what each exercises. The scheduler: test_srdsim_schedule, model_srdsim_schedule_exhaustive, test_srdsim_adversarial and test_srdsim_liveness. The transport: test_srdsim_transport, model_srdsim_framing, test_srdsim_adversarial, test_srdsim_get and test_srdsim_liveness. The send queue: test_srdsim_async. The mapping: test_srdsim_mapping. The plugin with NCCL: CI's plugin builds and exports, and tests/gpu run by hand on a GPU. Host tests also run under ASan, UBSan and TSan.](images/testing-map.svg)

| Test | What it establishes |
|---|---|
| `test_srdsim_schedule` | For each of the seven modes over 64 seeds: every submitted byte is released exactly once and to the right place; the six modes that keep I1 never release a signal early, while `violate-i1` sometimes does; each mode's defining order holds; a seed reproduces its order. Over 60,000 seeds, the two-operation case reaches all 2,520 legal orders and no illegal one |
| `model_srdsim_schedule_exhaustive` | Every state the scheduler can reach, for configurations small enough: in `srd` and `inorder` none releases a signal before its put, in `violate-i1` at least one does, every unit is released exactly once, and every path ends |
| `test_srdsim_transport` | Over a real TCP connection: every byte and signal arrives exactly once, the order of application equals the order of release, and a reader that acquire-loads a signal word sees the bytes placed before it (T12) |
| `model_srdsim_framing` | Every sequence of frame kinds and outcomes: after each frame the receiver is at the start of the next one or the link is closed |
| `test_srdsim_adversarial` | A hostile peer and misuse: an enormous frame length, an overflowing `offset + length`, an operation id never issued, a zero chunk size, a shutdown racing a sender. Each is refused or survived |
| `test_srdsim_async` | The send queue keeps per-peer order whatever order the copies complete in, and an operation stays incomplete until its last unit has left |
| `test_srdsim_get` | Get requests and replies carry the right source, destination and owner |
| `test_srdsim_mapping` | The BAR mapping and staging-pool arithmetic, at every slot and across page offsets, including values chosen to overflow naive arithmetic |
| `test_srdsim_liveness` | Bounds on elapsed time under impairment: slow is told apart from never |

CI also runs these under AddressSanitizer, UBSan and ThreadSanitizer; builds the plugin against the headers of each supported NCCL release in each delivery configuration, and with CUDA 12 and 13 with and without GDRCopy; checks that the op tables NCCL looks up are exported; and loads the host-only build with `dlopen`. The op table, connection setup, the delivery contexts, completion and gets are exercised end to end only by a run under NCCL on a GPU ([#8](https://github.com/littlemex/nccl-srdsim/issues/8)). The strong-signal hold has no host test yet.

## What is not simulated

![Reproduced: RMA through NCCL's GIN proxy backend; every delivery order the weak-signal contract allows; orders that break it on purpose; puts, signals, strong signals and gets; host and device windows. Not reproduced: collectives, which use the net plugin; device-side GIN (GDAKI, EFA-GDA), which has no host on the path; NIC queues, doorbells and PCIe order, which are hardware; loss and duplication, since SRD is reliable; bandwidth and latency, since data goes through host memory.](images/scope.svg)

| Not simulated | Why |
|---|---|
| NCCL's collectives (all-reduce, all-gather, send and receive) | They use the network plugin interface, which this plugin does not implement |
| Device-side GIN backends (GDAKI, EFA-GDA) and anything built on NVSHMEM | The GPU drives the NIC directly; no host op table is on the path |
| Work-request encoding, doorbells, completion-queue limits, PCIe and MMIO ordering | These are properties of the device path and the hardware memory model |
| Loss and duplication | SRD is reliable |
| Bandwidth, latency, anything measured in seconds | Payloads are staged through host memory; the simulator makes no performance claim |

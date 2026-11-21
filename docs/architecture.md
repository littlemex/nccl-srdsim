# Architecture

**English** | [日本語](architecture.ja.md)

nccl-srdsim (the plugin, or the simulator: the same thing) is an NCCL plugin that delivers GPU-initiated writes in any order an unordered fabric such as SRD may produce. It lets code written for that fabric show, on an ordinary GPU host, the failures it would otherwise show only there.

## Background

A **rank** is one process of an NCCL job, usually driving one GPU, and a **communicator** is the group of ranks NCCL has connected. GIN (GPU-Initiated Networking) is the NCCL API with which a GPU kernel writes into another rank's memory (a **put**), reads from it (a **get**), or increments a 64-bit counter there (a **signal**) that code on that rank waits on. A put can carry a signal, so that the waiting code learns the put's data is ready. In this document the **target** is the rank a put is written to and a **peer** is any other rank of the communicator.

NCCL hands these operations to an **RMA plugin** (remote memory access), a host library that carries them between ranks. SRD (Scalable Reliable Datagram, the transport of AWS EFA) is **reliable** but **unordered**: every message arrives exactly once, in any order. An RMA plugin built on SRD guarantees only the orderings GIN's contract names (next section).

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
| Locally complete | What `test` reports. For a put or a signal, the operation's source may be reused, which says nothing about the target. For a get, its data is in the local window, with two exceptions that deliver no data: a dry-run get (`SRDSIM_DRY_RUN=1`) completes at once, and a get whose plugin context was destroyed before its reply landed is reported done with only a log line saying its data did not land |
| Window | A registered memory region. Units name a window by number; each rank resolves the number to its own address |
| Plugin context | What `createContext` returns: it holds the number of RMA contexts NCCL asked for, and one scheduler for all of them |
| RMA context | NCCL's unit of ordering on a communicator, named by the context index of each operation. The ordering the contract promises is per RMA context and target; nothing is promised across RMA contexts |
| Stream | The TCP connection carrying one direction between two ranks |

## Where the plugin sits

NCCL's GIN proxy backend (`NCCL_GIN_TYPE=2`) runs a host thread that takes each operation a GPU kernel posts and calls the RMA plugin's functions: `iput`, `iputSignal`, `iget`, `iflush`, `test` and `rmaProgress`. This plugin provides that table of functions (the op table). Its receiving half, not a NIC, performs every write into the target's memory, so the plugin chooses the order in which writes take effect.

![Rank A, sending: the GPU kernel's GIN calls go through NCCL's GIN proxy backend to this plugin's op table, whose scheduler and send queue release units onto one TCP stream per peer. Rank B, the target: the receiving agent writes each unit, in release order, into the data window or the signal window of its own memory.](images/architecture-overview.svg)

| Part | What it does |
|---|---|
| The application's GPU kernel | Issues puts, signals and gets through NCCL's GIN device API, and waits on signals |
| NCCL's GIN proxy backend | A host thread that turns each operation into a call to the plugin, and calls `rmaProgress` and `test` |
| This plugin, sending half | Splits each operation into units, chooses the order of release, and sends the units to the target |
| This plugin, receiving half (its receiving agent) | On the target, writes each unit into the target's own memory, in the order it was released |

Every rank runs the sending half and the receiving half: it sends its own operations and applies its peers'. What this means for code that will run on a different GIN backend in production is in [What is not reproduced](#what-is-not-reproduced).

## Life of a signalled put

A signalled put goes through six steps; the numbers match the figure.

![Six steps in a row: 1 submit (iputSignal returns at once), 2 split (8 KiB placements plus one signal unit), 3 release (rmaProgress pulls up to 4 MiB per call, over all plugin contexts), 4 send (in release order, on the target's stream), 5 apply (the target writes its own memory), 6 complete (test: the source may be reused; iflush: everything released and every get landed). The sender's proxy thread runs steps 1 to 4 and 6; the target's receiving agent runs step 5.](images/put-lifecycle.svg)

1. **Submit.** NCCL's proxy thread calls `iputSignal` with the source, the size, the destination, the target rank and the signal. The plugin turns each memory handle (what it returned when the window was registered) into a window number, submits an operation, tagged with its RMA context index, to the scheduler of the plugin context it was called on, and returns at once.
2. **Split.** The scheduler splits the payload into `ceil(size / chunk)` placement units (`SRDSIM_CHUNK_BYTES`, 8 KiB by default) and adds one signal unit.
3. **Release.** Each `rmaProgress` call asks the scheduler of every plugin context on the communicator for units, up to 4 MiB over all of them, counting each unit's payload and frame header (a unit that is the first of the call always goes, and once one does not fit the call releases nothing more), starting from a different context on each call. Which units may go, and in what order, is the delivery mode (next section). A unit is released only with room reserved for it in its target's outbox, so a released unit is never refused for lack of room; an operation to a target whose outbox is full stays pending, and operations to other targets are released as before. The randomised modes draw from a generator seeded with `SRDSIM_SEED`, so the same submissions and progress calls give the same order of release.
4. **Send.** Released units are written to the target's stream in release order. Within one `rmaProgress` call, the frames of the units that need no queued copy are appended to their targets' outboxes, and each outbox that received one is pushed to its socket once afterwards. A push that fails means the link broke: every unit of the batch sent on it is reported as a failed send, which `queryLastError` then reports. Bytes an earlier call left in an outbox are pushed at the start of the next call; a failure of that push is not reported there. If a unit's source is device memory, the unit is first copied to pinned host memory, and the send queue sends no later unit to that target until the copy has finished, so the copy's timing cannot change the order the scheduler chose. (With GDRCopy, a library through which the CPU reads and writes GPU memory directly, or with `SRDSIM_ASYNC_DELIVERY=0`, each copy is waited for instead of queued.)
5. **Apply.** On the target, the receiving agent reads each frame (one unit as sent on the stream), resolves the window number to its own address, checks the bounds, and writes the bytes or adds to the signal word.
6. **Complete.** `test` reports a put or a signal done once all its units have been released and none is still in the send queue: the source may be reused, but nothing is known about the target. It reports a get done once every byte of its reply has been applied in the local window. `iflush` completes when the plugin context's scheduler has released everything submitted to it, none of its units is still in the send queue, and every byte of every get it issued has been applied. Its context index, memory handle and rank are not used, and the condition is evaluated each time `test` is called on the flush. It therefore waits for every RMA context and every target of the plugin context, and for operations submitted after the flush, which fences more than the interface requires; an operation held because its target's outbox is full holds every flush of the plugin context. It does not wait for puts to be applied at the target.

![A 32 KiB signalled put becomes placements P0 to P3 of 8 KiB and one signal S. Under srd the order P2, P0, P3, P1, S is legal because S comes after the last placement. Under violate-i1, S goes before every placement, which breaks I1. The target applies units in exactly the release order.](images/put-to-units.svg)

A TCP stream is reliable and ordered, so it adds no reordering of its own, and each receiving thread issues its writes in the order its frames arrive: a host write completes before the next frame is read, and device writes are queued on one CUDA stream (CUDA's in-order work queue, not a TCP stream), or are waited for, in that order. The order of release on one RMA context is therefore the order in which that context's writes take effect at the target. What a seed does not fix is the interleaving between RMA contexts and between senders.

## The delivery contract and the modes

The scheduler follows one rule: reorder wherever the contract promises no order, and keep every order it does promise. Reordering less would hide the orderings the simulator exists to produce; reordering more would report failures that no conforming fabric can cause. The contract is:

- **I1.** A weak signal becomes visible only after every byte of its own put has been applied. Nothing is promised about earlier puts or signals.
- A **strong** signal also becomes visible only after every earlier put and signal of the same RMA context to the same target has been applied (see [Signals](#signals)).
- Puts are applied in any order relative to each other, also when they go to the same target, and the chunks of one put are applied in any order.

Choosing a unit costs O(pending operations + log chunks): the scheduler counts what each operation can release and finds the drawn chunk in a tree over that operation's unplaced chunks, so a long put does not make every later choice slower. Completion and the duplicate-id check look an id up. The options, their order and the random draws are those of the list-building choice they replace; `test_srdsim_schedule` checks this in S12 and S13 (see [Testing](#testing)).

The scheduler chooses among the operations pending in a plugin context, on all its RMA contexts, when NCCL calls `rmaProgress`. An operation that NCCL submits and the plugin releases before the next one is submitted cannot be overtaken, so the orders a run reaches depend on how NCCL calls the plugin as well as on the mode.

![Two operations to the same target: A with placements A0 and A1 and no signal, submitted first, and B with placements B0 to B2 and signal Bs. One row per mode shows an example order. srd: random per seed, and Bs before A0 is legal. inorder: submission order. peer-seq: A before B, chunks reordered. reverse-ops: newest operation first. signal-last: every payload, then the signals. defer-one: the oldest operation held to the end. violate-i1: a pending signal goes first, which breaks I1.](images/modes.svg)

`SRDSIM_MODE` selects the mode:

| Mode | Keeps I1 | What it is for |
|---|---|---|
| `srd` (default) | yes | A random legal order per seed |
| `inorder` | yes | Submission order across all RMA contexts and targets of the plugin context. An operation whose target's outbox is full is passed over, so operations to other targets go ahead of it. The control: code that fails here fails for a reason that has nothing to do with order |
| `peer-seq` | yes | Submission order per RMA context and target, chunks within an operation reordered. More order than the contract promises, as aws-ofi-nccl (the production RMA plugin for SRD) happens to keep. Code that passes here but fails under `srd` relies on more than the contract |
| `reverse-ops` | yes | Newest operation of the plugin context first, lowest chunk first: the simplest order that is not submission order |
| `signal-last` | yes | Every payload of every pending operation of the plugin context, then the signals: payload and signal as far apart as the contract allows. Payloads held for a full outbox do not hold the signals of other targets |
| `defer-one` | yes | The oldest pending operation of the plugin context held back until every other has been released, as when one path is slow; chunk order is random. It stays held while another operation is only waiting for room |
| `violate-i1` | no | Releases a pending signal before anything else, whether or not its put's placements have been released, and ignores the strong-signal hold. A negative control |

## Threads, RMA contexts and streams

Each rank runs three kinds of thread for a communicator: NCCL's GIN proxy thread submits and progresses operations, the plugin's receiving agent runs one thread per inbound stream that applies them, and the plugin's responder thread sends the replies to gets.

![Rank 0, one communicator: NCCL's GIN proxy thread calls iput, iputSignal, iget, rmaProgress and test on two RMA contexts, ordered per target in one shared scheduler; nothing is ordered across them. Both feed one send stream per peer. The receiving agent runs one thread per inbound stream, and a responder thread sends the replies to gets. Ranks 1 and 2 have the same structure; every pair of ranks has a stream in each direction, and traffic to rank 0 itself goes through a socket pair along the same path.](images/threads.svg)

- **Threads.** NCCL's GIN proxy thread calls the submit functions, `rmaProgress` and `test`. On the target, the receiving agent (one per communicator) reads and applies each inbound stream on a thread of its own. When a frame is a get request, that thread copies the requested bytes and hands them to the communicator's responder thread, which sends the reply.
- **RMA contexts.** `createContext` makes a plugin context with the number of RMA contexts NCCL asks for. Its operations share one scheduler and mutex, and the scheduler keeps their ordering per RMA context and target.
- **Streams.** In `listen`, each rank opens a TCP listening socket on all interfaces and returns, as the handle NCCL exchanges, the address its peers should dial: `SRDSIM_LOCAL_IP` if set, otherwise the address of the default route. In `connect`, each rank dials every peer and accepts a stream from every peer, so every pair of ranks has a stream in each direction. A rank's traffic to itself goes over a socket pair through the same path, because a NIC also delivers, and may reorder, writes a rank sends to itself. Sockets are IPv4, and both ends must be the same build on the same architecture, since frames are not converted between byte orders.

## Components

The plugin is one source file and four headers it includes; a fifth header is used only by a test. The headers hold the logic whose correctness can be decided without a GPU, so they are tested on any Linux host.

![nccl_rma_srdsim.cc includes srdsim_schedule.h (order of release), srdsim_transport.h (frames and the receiving agent), srdsim_async.h (the send queue) and srdsim_mapping.h (BAR arithmetic), and uses NCCL's plugin headers and, optionally, the CUDA driver with NVRTC and GDRCopy. The headers need only the C++ standard library, and POSIX sockets for the transport. The plugin also includes srdsim_impair.h (impairments), which the liveness test uses too.](images/components.svg)

| File | Responsibility |
|---|---|
| `src/srdsim_schedule.h` | The scheduler: splits each operation into units and decides the order of release according to the delivery mode |
| `src/srdsim_transport.h` | The transport: the frame format, the streams, the outbox per peer, and the receiving agent |
| `src/srdsim_async.h` | The send queue for units whose source is device memory still being copied to the host. A template over the unit type |
| `src/srdsim_mapping.h` | Address arithmetic for GDRCopy BAR mappings and the pinned staging pool |
| `src/srdsim_impair.h` | Deliberate impairments (delay, bursts, a slow receiver, a small send buffer, a frozen link, a lagging signal), read from the environment by each communicator and used by the liveness test |
| `src/nccl_rma_srdsim.cc` | The op table, stream setup, window registration, delivery into device memory, completion and diagnostics |

## The transport and the receive side

The sending half never writes into the target's memory. It sends a frame header and the bytes over the stream, and the target's receiving agent writes into its own window. This way, the plugin needs neither GPU peer access between processes nor exporting an allocation across a process boundary, and it works the same within one host and across hosts.

![The receiving agent reads a frame header from the inbound stream, resolves the window number to its own address in the window table, checks that the window is known and the write is within bounds, then writes a host window with memcpy, a device window with a CUDA copy or GDRCopy, or adds to a signal word. A refused frame is skipped; if skipping would lose the start of the next frame, the stream is closed.](images/receive-side.svg)

A unit names a window by number, not by address. Window registration (`regMrSym`) is done by all ranks together, in the same order, so the numbers agree on every rank as long as every rank registers the same windows; each target resolves a number to its own base address.

Every frame is a 56-byte header followed by `length` bytes of payload. `length` always counts the bytes that follow, whatever the kind, which is what lets the receiving agent find the next frame after refusing one. A get request carries no payload, so its `length` is 0 and the number of bytes to read travels in `value`.

![The 56-byte header: kind, window, dest_window and requester (4 bytes each), then offset, length, value, op_id and dest_offset (8 bytes each), followed by length bytes of payload. Placement writes length bytes at window plus offset; Signal adds value to the 8 bytes at window plus offset; GetRequest reads value bytes (length is 0) and replies to requester at dest_window; GetResponse is a placement counted for the requester's get whose id is op_id; GetError (length 0) says the get whose id is op_id will not be served, and the requester fails it; Shutdown closes the stream. dest_window, requester and dest_offset are used by gets only.](images/wire-frame.svg)

| Layer | Responsible for | Retransmission and failure | Identifies |
|---|---|---|---|
| Stream (TCP) | Reliable, ordered bytes from one rank to another | TCP retransmits; a broken stream fails the plugin contexts sending on it | The pair of ranks and the direction |
| Outbox | Bytes appended but not yet written to the stream, up to 64 MiB per peer counting the room reserved for units being released. Every send pushes as much as the socket takes; `rmaProgress` and the responder push the rest | A unit the scheduler releases holds reserved room and is never refused. Any other frame (a get reply) is refused when it would not fit beside what is queued and reserved, and its sender retries. A frame larger than the whole outbox is accepted when the outbox is empty | The peer |
| Frame | One unit: its kind, where it goes, its length | Not retransmitted. A frame naming an unknown window or running past its window is skipped, and a skipped or unwritten placement withholds its operation's signal; one longer than the receiver's frame limit, `max(SRDSIM_CHUNK_BYTES, 4 MiB)`, or of an unknown kind, closes the stream. `SRDSIM_CHUNK_BYTES` is at most 64 MiB, and above 4 MiB every rank must use the same value, or a larger frame closes the link | `window` and `offset` locate the write. A placement and a signal carry their operation's id in `op_id`, unique in the sending process; for a get reply and a GetError, `op_id` names the requester's get |
| Window table | Turning a window number into this rank's address | A number no longer registered is refused | `window` |

A skipped placement is lost, and its operation's signal is then withheld: the receiving thread records, by the id the frames carry, each operation that lost a placement, and does not apply that operation's signal, which would announce bytes that never landed. It logs and counts the withheld signal apart from refused frames; other operations on the link are unaffected. Past 65,536 such operations on one link, the link is closed. This relies on I1: a signal follows every placement of its operation. In `violate-i1`, which releases a signal first on purpose, the signal may already be applied when a placement of its operation is skipped. A get reply that will not come no longer leaves the get waiting in these cases. When the owner cannot serve a request (an unknown or deregistered window, a read past the window, or a failed device read), it answers with a GetError frame. While its communicator is closing, it sends the GetError directly, once, and only logs it if that send fails; a request naming a rank outside the communicator is logged and not answered. when the requester cannot write a frame of a reply (an unknown or deregistered destination, a frame past it, or a failed write), it fails the get itself. Either way only the bytes that will not arrive are counted as arrived, so frames still on their way land and are counted before the get or a flush completes; `test` then reports the get done with a log line saying its data did not land, and `queryLastError` reports the failure. Still open ([#37](https://github.com/littlemex/nccl-srdsim/issues/37)): a link that closes while replies are outstanding, and checking that a GetError comes from the rank the get was sent to.

## Delivery into device memory

The hazard on a GPU is a deadlock: the application's kernels stay resident on the GPU, busy-waiting for data, while the copy that delivers the data is queued behind a module load or a free in the application, which holds its CUDA context's lock until those kernels finish. The four ways of writing a device window differ in what a copy can be queued behind.

![Four columns. Own CUDA context (the default): a CUDA copy from a context the plugin made; needs VMM windows; does not wait for the application's module loads. Primary (SRDSIM_DELIVERY_CONTEXT): a CUDA copy from the primary context; accepts any allocation; waits behind module loads and frees and can deadlock. Green: a CUDA copy on the smallest SM partition; needs VMM windows and CUDA 12.4; no time-slice wait, but shares the primary context's lock. GDRCopy (build option): the CPU writes through the GPU's BAR mapping; needs /dev/gdrdrv and VMM windows; never waits for the GPU.](images/delivery-contexts.svg)

| Way | What it does | What it needs | What a write can wait behind |
|---|---|---|---|
| Own CUDA context (default) | Copies from a CUDA context the plugin creates for itself | Device windows allocated with `ncclMemAlloc` (VMM allocations, whose mappings belong to the process rather than to one CUDA context); registration refuses any other | Nothing the application does on the application's CUDA context, such as loading a module while its kernels wait (DeepEP's JIT does); but each copy waits for a time slice on the GPU while kernels are resident, so copies are slow |
| Primary (`SRDSIM_DELIVERY_CONTEXT=primary`) | Copies from the device's primary CUDA context (the one CUDA creates by default and most applications use) | Any allocation | The application's module loads and frees on that context: the deadlock above. It exists to reproduce it |
| Green (`SRDSIM_DELIVERY_CONTEXT=green`) | Copies from a green context, a CUDA context limited to the smallest group of SMs (streaming multiprocessors) the device allows | VMM windows and CUDA 12.4 or later | Not a time slice on the GPU, but the primary context's lock, which a module load holds; suits applications that load and free nothing while their kernels wait |
| GDRCopy (build option) | Maps the window into the process through the GPU's BAR (its PCIe memory aperture), and the CPU writes it | `/dev/gdrdrv`, and VMM windows unless the context is `primary`. A window not aligned to a GPU page falls back to a CUDA copy that is waited for | Nothing on the GPU |

## Signals

A signal unit adds to a 64-bit word, the signal word, in the target's signal window: the signal's value for `NCCL_GIN_SIGNAL_OP_ADD`, and one otherwise.

![Three panels. Host window: the receiving agent writes the payload, then adds to the signal word with memory_order_release, which pairs with the reader's memory_order_acquire load, after which the reader sees every payload byte; T12 checks this under ThreadSanitizer. Device window: in a window registered with SIGNAL_NEVER_RESET the plugin stores the running total it keeps, which needs no SM and is valid only because the plugin is the only writer; in any other window an atomic add runs in a kernel compiled with NVRTC. Strong signal: S! is held on its RMA context and target until A and B are released.](images/signals.svg)

- **Host windows.** The agent adds to the word atomically with `memory_order_release`, after the payload bytes it placed from the same stream. A reader that loads the word with `memory_order_acquire`, as the application's waiting code does, then sees those bytes. The transport test T12 is such a reader; with the add made relaxed, ThreadSanitizer reports it.
- **Device windows registered with `NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET`.** The plugin keeps the running total of each signal word, starting from zero, and stores it as eight aligned bytes. The flag says the application never resets these words, as NCCL does for the signal window of its GIN proxy when the backend supports it ([`gin_host_proxy.cc`](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/gin/gin_host_proxy.cc#L578)), and from that NCCL and this plugin take the plugin to be the only writer. A store of the total then equals an atomic add, given that the word holds zero when the window is registered and that totals are taken and stored in order; the plugin computes and stores each total under one lock. No kernel is launched, so delivery needs no SM while the application's kernels occupy them all.
- **Any other device window.** The application may reset a word itself, so the add is applied by a small kernel, compiled with NVRTC when the first device window is registered and launched on the stream that carries the operation's placements. `SRDSIM_SIGNAL_KERNEL=1` uses the kernel for every device window, so that the two ways can be compared. Deregistering a window waits for its pending adds.
- **Strong signals.** SRD has no strong signal, but NCCL may request one, and a submit call that returns an error stops NCCL's GIN proxy thread for good. So the plugin accepts strong signals, and the scheduler holds each one until every earlier operation of that RMA context to that target has been released. Operations on other RMA contexts do not hold it. In the figure, S! waits for the earlier operations A and B. With NCCL headers that define the v16 op table, the plugin also reports `flushesAllPutsOnAnySignal` as false: a signal does not imply that puts from other peers have been applied.

## Gets

A get is one-sided, as an RDMA read is: the owner's plugin answers it without the owner's NCCL progress thread.

![1: iget submits a GetRequest unit, which is released like any unit. 2: the owner's receive thread copies the requested bytes. 3: the owner's responder thread sends them in chunks. 4: the GetResponse chunks travel back in order, each naming the get. 5: the requester counts the bytes applied per get, and the get completes when none is left.](images/get-flow.svg)

- **Request.** `iget` records the get under an id the process never reuses, with the number of bytes still to land, and submits a GetRequest unit carrying that id. A get of zero bytes completes at once and sends no request.
- **Reply.** On the owner, the receive thread that reads the request copies the bytes from the window and hands them to the communicator's responder thread. The responder sends them in frames of `SRDSIM_CHUNK_BYTES`, but at most 4 MiB, the frame every receiver accepts whatever its own chunk size, in the order the requests arrived from each requester. It serves requesters in turn, one reply at a time, and its sends never block: a frame the outbox refuses is retried on a later turn, so a requester that stops reading holds up only its own replies. The copies queued for a requester are not bounded. The reply does not go through the owner's scheduler, so the delivery modes do not reorder its chunks. When the communicator closes, the responder sends what it can for up to one second, then logs the replies it did not send; a request that arrives after that is refused.
- **Completion.** Each reply frame names the get by its id. The requester subtracts the frame's length from the get, and from the total its plugin context's `iflush` waits on; `test` reports the get done when no byte is left. A reply for an id with no record is dropped. A get the owner refuses, or whose reply cannot be written here, fails as described under [the transport](#the-transport-and-the-receive-side); the owner sends its GetError through the responder, behind its earlier replies to that requester, except while the communicator is closing, when it sends it directly. When a plugin context is destroyed, its outstanding gets are aborted: a late reply is dropped, and `test` reports the get done, with a log line saying its data did not land.

EFA-GDA, the device-side GIN backend for SRD, has no get; this path exists for code written against GIN backends that have one, such as aws-ofi-nccl's host-proxy backend, whose `iget` is an [`fi_read`](https://github.com/aws/aws-ofi-nccl/blob/2475c4c736907e68ebeadb3848f31231ee4a44d0/include/rdma/gin/nccl_ofi_gin_reqs.h#L414).

## Impairments and diagnostics

`src/srdsim_impair.h` models a slow, bursty, congested or partly frozen fabric, never a lossy one. Each communicator reads the impairments from the environment when it connects (see the configuration reference): delays, bursts and a signal lag on the progress thread before each released unit, a wait before each frame the receive agent applies, the socket send buffer, and a link to one peer frozen after a number of units by holding releases to it. The liveness test sets the same impairments directly. Such impairments target liveness failures, where something stops making progress, which are often the hardest defects to find on real hardware; ordering failures, where some reachable state is wrong, are what the delivery modes target.

Besides the lines it prints at registration, deregistration and teardown, the plugin writes two kinds of line to stderr whatever `NCCL_DEBUG` is. In a CUDA build, when the first device window is registered, one line names the mechanism that writes device windows (GDRCopy, queued CUDA copies or waited-for CUDA copies), the CUDA context copies run from, and how signals are applied. Each time a communicator closes, one line reports how many times a progress call held a peer back for want of outbox room (counted once per peer per call), and how many frames its outbox refused for its budget.

For diagnosis, `SRDSIM_LOG_OPS=1` logs every operation and also starts a stall monitor, which reports where each thread last was when no unit has been applied for `SRDSIM_STALL_MS`. Every setting is listed in [the configuration reference](configuration.md).

## Testing

The properties that can be decided without a GPU are tested on every pull request: on the headers, and on the built plugin through its op table. The plugin runs under NCCL only in the GPU test, which is run by hand.

![A table of tests against what each exercises. The scheduler: test_srdsim_schedule, model_srdsim_schedule_exhaustive, test_srdsim_adversarial and test_srdsim_liveness. The transport: test_srdsim_transport, model_srdsim_framing, test_srdsim_adversarial, test_srdsim_get and test_srdsim_liveness. The send queue: test_srdsim_async. The mapping: test_srdsim_mapping. The plugin op table: test_plugin, which also exercises the scheduler, the transport and the send queue. The plugin with NCCL: CI's plugin builds and exports, and tests/gpu run by hand on a GPU. Host tests and the plugin harness also run under ASan, UBSan and TSan.](images/testing-map.svg)

| Test | What it establishes |
|---|---|
| `test_srdsim_schedule` | For each of the seven modes over 64 seeds: every submitted byte is released exactly once and to the right place; the six modes that keep I1 never release a signal early, and `violate-i1` does; each mode's defining order holds; a seed reproduces its order. Over 60,000 seeds, a case of two operations reaches all 2,520 legal orders and no illegal one. S12 pins the order of release itself: a scripted workload's released sequence is hashed in every mode and several seeds and compared with a table recorded from the list-building choice, so a change meant to keep the order must leave every hash unchanged. S13 drives the scheduler and a reference scheduler that chooses by building the full list of options through the same random workloads, with operations of up to 300 chunks and links full for some peers, and requires the identical sequence of units |
| `model_srdsim_schedule_exhaustive` | Every state the scheduler can reach, for configurations small enough: in `srd` and `inorder` none releases a signal before its put, in `violate-i1` at least one does, every unit is released exactly once, and every path ends |
| `test_srdsim_transport` | Over a real TCP connection: every byte and signal arrives exactly once, the order of application equals the order of release, and a reader that acquire-loads a signal word sees the bytes placed before it (T12) |
| `model_srdsim_framing` | Every sequence of the frame kinds and outcomes it models: after each frame the receiving agent is at the start of the next one or the stream is closed |
| `test_srdsim_adversarial` | A hostile peer and misuse: an enormous frame length, an overflowing `offset + length`, an operation id never issued, a zero chunk size, a shutdown racing a sender. Each is refused or survived |
| `test_srdsim_async` | The send queue keeps per-target order whatever order the copies complete in, and an operation stays incomplete until its last unit has left |
| `test_srdsim_get` | Get requests and replies carry the right source, destination and owner |
| `test_srdsim_mapping` | The BAR mapping and staging-pool arithmetic, at every slot and across page offsets, including values chosen to overflow naive arithmetic |
| `test_srdsim_liveness` | Bounds on elapsed time under impairment: slow is told apart from never |
| `test_plugin` (`tests/plugin`) | The built plugin, loaded with `dlopen` and called through one op-table version at a time as NCCL's GIN proxy calls it: a proxy thread that submits in batches with NCCL's aggregation rule, tests in issue order and retires at the per-target frontier, and overwrites a put's source when it retires. One process per rank. Cases cover I1 and its negative control, strong signals and their control, RMA contexts, gets and flushes with the owner stopped or idle, backpressure to a stopped target, a context rebuilt with a get outstanding, refused and dry-run requests, a get the owner cannot serve, a signal whose operation lost a placement, frames above 4 MiB, and registration during traffic. With a GPU, three cases check how a signal is applied to a device window with and without `NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET`, and that a put's bytes are in a device window when its signal becomes visible; without one they are skipped |

CI also runs the host tests under AddressSanitizer, UBSan and ThreadSanitizer; builds the plugin against the headers of each supported NCCL release in each delivery configuration, and with CUDA 12 and 13 with and without GDRCopy; checks that the op tables NCCL looks up are exported; and loads the host-only build with `dlopen`. It runs `test_plugin` for every op-table version the plugin exports, in the host-only build, under each sanitizer, and in the CUDA build on a runner that has the driver library but no GPU. Delivery into device windows through NCCL is exercised only by the GPU test ([#8](https://github.com/littlemex/nccl-srdsim/issues/8)).

## What is not reproduced

The simulator reproduces the order of delivery through NCCL's GIN proxy backend, and nothing below it.

![Reproduced: RMA through NCCL's GIN proxy backend; contract-legal orders among pending operations; orders that break it on purpose; puts, signals, strong signals and gets; host and device windows. Not reproduced: collectives, which use the network plugin interface; device-side GIN (GDAKI, EFA-GDA), which has no host on the path; NIC queues, doorbells and PCIe order, which are hardware; loss and duplication, since SRD is reliable; bandwidth and latency, since data goes through host memory.](images/scope.svg)

| Not reproduced | Why |
|---|---|
| NCCL's collectives (all-reduce, all-gather, send and receive) | They use the network plugin interface, which this plugin does not implement |
| Device-side GIN backends (GDAKI for InfiniBand and RoCE, EFA-GDA for SRD) and anything built on NVSHMEM | The GPU drives the NIC directly; no host op table is on the path. Code that will use one in production is tested here against the contract that backend promises, not on that backend |
| What the fabric can advertise | NCCL's proxy backend reports strong signals as supported for every RMA plugin, so the plugin cannot say that SRD lacks them; it implements strong signals instead |
| Work-request encoding, doorbells, completion-queue limits, PCIe and MMIO ordering | These are properties of the device path and the hardware memory model |
| Loss and duplication | SRD is reliable |
| Bandwidth, latency, anything measured in seconds | Payloads are staged through host memory; the simulator makes no performance claim |

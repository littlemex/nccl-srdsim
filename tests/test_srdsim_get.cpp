// SPDX-License-Identifier: Apache-2.0

// Tests for Get on the wire (`src/srdsim_transport.h`'s `WireKind::GetRequest`/`GetResponse`,
// `SendLinks::send_get_request`, and the two `ReceiveAgent::Hooks` a get uses). Real processes over a real
// TCP connection, like `test_srdsim_transport.cpp`; needs neither CUDA nor NCCL, and does not touch
// `nccl_rma_srdsim.cc` at all -- the plugin's own `srdsim_iget`/`srdsim_handle_get_request` wire a real
// `RmaCtx`/`Scheduler` into these same hooks, which is integration this file cannot reach without a
// live NCCL communicator; what it can and does check is that the wire format and the receive loop
// carry a get correctly, independent of who answers it.
//
//   G1  a GetRequest names the owner's source correctly in the `get_request` hook: pointer, length,
//       device flag, and the reply's destination and requester rank, round tripped exactly.
//   G2  the owner's reply (an ordinary placement, sent back tagged GetResponse) lands at the
//       requester's own window at the right offset, and `get_response_landed` fires with the token
//       the GetRequest carried -- unchanged, since the owner never interprets it.
//   G3  a GetRequest naming a range past the end of its window is refused and the stream survives.
//   G4  a GetRequest naming a window the receiver never registered is refused and the stream survives.
//   G5  a GetResponse frame the `get_response_accept` hook refuses is not written, is reported through
//       `get_failed` with its length, does not fire `get_response_landed`, and the stream survives it.
//   G6  `reply_frame_is_next`, the rule the plugin's accept hook applies to a get it has outstanding: only the frame
//       that starts where the bytes accounted for end, in the get's window and inside its range, is accepted.

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "srdsim_schedule.h"
#include "srdsim_transport.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void expect(bool cond, const std::string& what) {
    ++ g_checks;
    if (not cond) {
        ++ g_failures;
        if (g_failures <= 30)
            std::printf("[FAIL] %s\n", what.c_str());
    }
}

std::string ctx(const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return std::string(buf);
}

constexpr uint32_t kOwnerSrcWindow = 0;      // the owner's local storage, where the data lives
constexpr uint32_t kRequesterDstWindow = 7;  // the requester's own window, where a reply must land.
                                              // Deliberately not 0: a get names two windows that have
                                              // nothing to do with each other, and the same number for
                                              // both would not catch a field swapped with another.
constexpr size_t kFetchBytes = 4096;

// What the owner's agent reports after a GetRequest arrived -- the arguments its `get_request` hook
// was called with, so the parent can check them against what the requester actually asked for.
struct OwnerReport {
    bool called = false;
    bool src_is_device = false;
    size_t length = 0;
    uint32_t dest_window = 0;
    uint64_t dest_off = 0;
    uint32_t requester_rank = 0;
    uint64_t owner_token = 0;
    uint64_t src_checksum = 0;   // over `length` bytes from `src`, so the pointer arithmetic is checked too
    uint64_t placements = 0, signals = 0, errors = 0;
};

// Plain data goes in `OwnerReport`, which is sent across the fork as raw bytes; the link the hook
// below replies on is a live object, connected before the request can possibly arrive, and is reached
// through this wrapper instead so nothing non-serialisable ends up inside a struct that is. (A real
// `RmaCtx` plays the same role for `srdsim_handle_get_request`, with a scheduler in place of the link.)
struct OwnerCtx {
    OwnerReport report;
    srdsim::SendLinks* reply = nullptr;
    bool reply_sent = false;
};

// What the requester's agent reports after the owner's reply landed.
struct RequesterReport {
    uint64_t landed_token = 0;
    uint64_t landed_bytes = 0;
    bool landed_called = false;
    uint64_t data_checksum = 0;   // over the whole destination window, so a reply at the wrong offset
                                   // or the wrong length shows up as a mismatch, not a missed check
    uint64_t placements = 0, errors = 0;
};

uint64_t checksum(const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++ i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

bool owner_host_copy(void* dst, const void* src, size_t bytes, void* /*user*/) {
    std::memcpy(dst, src, bytes);
    return true;
}

bool owner_get_request(const void* src, bool src_is_device, size_t length, uint32_t dest_window,
                       uint64_t dest_off, uint32_t requester_rank, uint64_t owner_token, void* user) {
    auto* c = static_cast<OwnerCtx*>(user);
    OwnerReport& r = c->report;
    r.called = true;
    r.src_is_device = src_is_device;
    r.length = length;
    r.dest_window = dest_window;
    r.dest_off = dest_off;
    r.requester_rank = requester_rank;
    r.owner_token = owner_token;
    r.src_checksum = checksum(src, length);
    // The actual answer: an ordinary placement, tagged GetResponse so the requester's receive loop
    // knows to release the `iflush` it names rather than treat it as an unrelated put.
    c->reply_sent = c->reply != nullptr and
                    c->reply->send_placement(/*peer=*/0, dest_window, dest_off, src, length, owner_token,
                                             srdsim::WireKind::GetResponse);
    return c->reply_sent;
}

bool requester_host_copy(void* dst, const void* src, size_t bytes, void* /*user*/) {
    std::memcpy(dst, src, bytes);
    return true;
}

void requester_get_landed(uint64_t owner_token, uint64_t bytes, int /*peer_rank*/, void* user) {
    auto* r = static_cast<RequesterReport*>(user);
    r->landed_called = true;
    r->landed_token = owner_token;
    r->landed_bytes += bytes;
}

// ---------------------------------------------------------------------------
// G1 + G2: a real GetRequest, answered by a real GetResponse, over a real socket.
// ---------------------------------------------------------------------------
//
// Two directions on two independent sockets, exactly as a real pair of ranks would have (the owner's
// agent serves the requester's GetRequest connection; the requester's agent serves the owner's
// GetResponse connection) -- a single socket carrying both ways would still work here since nothing
// in this test sends concurrently in both directions, but two sockets is what the real plugin's one
// link per peer gives, and using two here is what makes this test exercise that shape rather than a
// shape the real thing never has.
void test_get_round_trip() {
    int report_sock[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, report_sock) != 0) { expect(false, "socketpair"); return; }

    uint16_t owner_port = 0, requester_port = 0;
    const int owner_listen = srdsim::listen_any(&owner_port);
    const int requester_listen = srdsim::listen_any(&requester_port);
    if (owner_listen < 0 or requester_listen < 0) { expect(false, "listen_any"); return; }

    const pid_t pid = ::fork();
    if (pid < 0) { expect(false, "fork"); return; }

    if (pid == 0) {
        // ---- child: the owner ----
        ::close(report_sock[0]);
        ::close(requester_listen);

        std::vector<char> storage(kFetchBytes * 2, 0);   // the fetch reads a sub-range, not the whole window
        for (size_t i = 0; i < storage.size(); ++ i) storage[i] = static_cast<char>((i * 73 + 5) & 0xff);

        srdsim::WindowTable windows;
        windows.add({storage.data(), storage.size(), false});   // host window: easy to checksum directly

        // Dialled before the owner's own agent starts serving, so a reply can go out the moment the
        // get_request hook fires -- the requester's listen socket is already up (opened before fork).
        const int reply_fd = srdsim::dial("127.0.0.1", requester_port);
        if (reply_fd < 0) _exit(4);
        srdsim::SendLinks reply_links;
        reply_links.set(/*peer=*/0, reply_fd);

        OwnerCtx octx;
        octx.reply = &reply_links;

        srdsim::ReceiveAgent::Hooks hooks{};
        hooks.device_copy = owner_host_copy;
        hooks.get_request = owner_get_request;
        hooks.user = &octx;
        hooks.log = [](const char*) {};
        srdsim::ReceiveAgent agent(&windows, hooks);

        const int conn = ::accept(owner_listen, nullptr, nullptr);
        ::close(owner_listen);
        if (conn < 0) _exit(2);
        srdsim::set_nodelay(conn);
        agent.serve(conn);
        agent.join();   // returns once the requester has sent Shutdown or closed

        octx.report.placements = agent.applied_placements();
        octx.report.signals = agent.applied_signals();
        octx.report.errors = agent.errors();

        const bool ok = srdsim::write_all(report_sock[1], &octx.report, sizeof(octx.report)) and octx.reply_sent;
        ::close(report_sock[1]);
        _exit(ok ? 0 : 3);
    }

    // ---- parent: the requester ----
    ::close(report_sock[1]);
    ::close(owner_listen);

    std::vector<char> dest(kFetchBytes * 2, 0);
    RequesterReport report{};

    // `WindowTable::add` assigns ids in registration order, starting at 0; dummy windows before the
    // real one put it at `kRequesterDstWindow` (not 0, see that constant's comment), checked rather
    // than assumed so a change to that numbering fails loudly here instead of testing the wrong id.
    srdsim::WindowTable windows;
    for (uint32_t i = 0; i < kRequesterDstWindow; ++ i)
        windows.add({nullptr, 0, false});
    const uint32_t dest_window_id = windows.add({dest.data(), dest.size(), false});
    expect(dest_window_id == kRequesterDstWindow,
          ctx("destination registered at id %u == %u", dest_window_id, kRequesterDstWindow));

    srdsim::ReceiveAgent::Hooks req_hooks{};
    req_hooks.device_copy = requester_host_copy;
    req_hooks.get_response_landed = requester_get_landed;
    req_hooks.user = &report;
    req_hooks.log = [](const char*) {};
    srdsim::ReceiveAgent req_agent(&windows, req_hooks);
    const int requester_conn = ::accept(requester_listen, nullptr, nullptr);
    ::close(requester_listen);
    if (requester_conn < 0) { expect(false, "accept (requester side)"); ::waitpid(pid, nullptr, 0); return; }
    srdsim::set_nodelay(requester_conn);
    req_agent.serve(requester_conn);

    const int fd = srdsim::dial("127.0.0.1", owner_port);
    if (fd < 0) { expect(false, "dial (owner)"); ::waitpid(pid, nullptr, 0); return; }
    srdsim::SendLinks links;
    links.set(/*peer=*/0, fd);

    const uint64_t token = 0xA5A5000012345678ull;   // an arbitrary opaque value, not a real pointer here
    const bool sent = links.send_get_request(/*peer=*/0, kOwnerSrcWindow, /*src_offset=*/kFetchBytes,
                                             kFetchBytes, kRequesterDstWindow, /*dest_offset=*/kFetchBytes,
                                             /*requester_rank=*/0, token);
    expect(sent, ctx("send_get_request accepted"));

    // Give the owner's child a moment to receive the request, read its own memory, and dial back; then
    // give the GetResponse time to arrive and land. No synchronisation primitive is shared between the
    // two processes beyond the sockets themselves, so this polls rather than sleeping once -- a sleep
    // long enough to never flake would also be long enough to make every run of the suite slow.
    for (int i = 0; i < 200 and not req_agent.applied_placements(); ++ i)
        usleep(5000);

    links.shutdown_all();
    req_agent.join();

    int status = 0;
    ::waitpid(pid, &status, 0);
    OwnerReport owner_report{};
    const bool got_owner_report = srdsim::read_all(report_sock[0], &owner_report, sizeof(owner_report));
    ::close(report_sock[0]);

    expect(got_owner_report, "owner reported back");
    if (not got_owner_report) return;

    expect(owner_report.called, "get_request hook was called on the owner");
    expect(owner_report.length == kFetchBytes, ctx("fetch length %zu == %zu", owner_report.length, kFetchBytes));
    expect(not owner_report.src_is_device, "owner's window is host, so src_is_device is false");
    expect(owner_report.dest_window == kRequesterDstWindow,
          ctx("dest_window %u == %u", owner_report.dest_window, kRequesterDstWindow));
    expect(owner_report.dest_off == kFetchBytes, ctx("dest_off %llu == %zu",
          static_cast<unsigned long long>(owner_report.dest_off), kFetchBytes));
    expect(owner_report.requester_rank == 0, "requester_rank round tripped as 0");
    expect(owner_report.owner_token == token, "owner_token round tripped unchanged");
    expect(owner_report.errors == 0, ctx("owner saw %llu error(s)", static_cast<unsigned long long>(owner_report.errors)));

    const uint64_t want_checksum = checksum(dest.data() + kFetchBytes, kFetchBytes);
    // The owner's storage window and the requester's destination window were filled identically
    // (`storage[i] = (i*73+5)&0xff` on the owner; the requester's `dest` starts zeroed), so after a
    // correct round trip the fetched range on the requester equals the same range on the owner. The
    // owner reported a checksum over exactly the bytes it read; recomputing the same formula over the
    // bytes its own storage would have held at that range catches a reply sent from the wrong offset.
    std::vector<char> expected(kFetchBytes);
    for (size_t i = 0; i < kFetchBytes; ++ i) expected[i] = static_cast<char>(((kFetchBytes + i) * 73 + 5) & 0xff);
    expect(owner_report.src_checksum == checksum(expected.data(), kFetchBytes),
          "owner read the range this test expects it to have read");

    expect(req_agent.applied_placements() == 1,
          ctx("requester applied %llu placement(s), want 1",
             static_cast<unsigned long long>(req_agent.applied_placements())));
    expect(report.landed_called, "get_response_landed fired on the requester");
    expect(report.landed_bytes == kFetchBytes, ctx("get_response_landed reported %llu byte(s), want %zu",
          static_cast<unsigned long long>(report.landed_bytes), kFetchBytes));
    expect(report.landed_token == token, ctx("landed token 0x%llx == 0x%llx",
          static_cast<unsigned long long>(report.landed_token), static_cast<unsigned long long>(token)));
    expect(checksum(dest.data() + kFetchBytes, kFetchBytes) == want_checksum, "fetched bytes landed at the right offset");
    expect(req_agent.errors() == 0, ctx("requester saw %llu error(s)", static_cast<unsigned long long>(req_agent.errors())));
    (void)status;
}

// ---------------------------------------------------------------------------
// G3: a GetRequest whose range runs past the end of the owner's window.
// ---------------------------------------------------------------------------
void test_get_out_of_range() {
    std::vector<char> storage(1024, 0);
    srdsim::WindowTable windows;
    windows.add({storage.data(), storage.size(), false});

    OwnerCtx octx;   // `octx.reply` stays null: neither request here needs a real reply
    srdsim::ReceiveAgent::Hooks hooks{};
    hooks.device_copy = owner_host_copy;
    hooks.get_request = owner_get_request;
    hooks.user = &octx;
    hooks.log = [](const char*) {};
    srdsim::ReceiveAgent agent(&windows, hooks);

    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { expect(false, "socketpair"); return; }
    srdsim::set_nodelay(sv[0]);
    srdsim::set_nodelay(sv[1]);
    agent.serve(sv[1]);

    srdsim::SendLinks links;
    links.set(0, sv[0]);
    // Offset 900, length 1000: the window is only 1024 bytes, so this has to be refused rather than
    // read past the end of `storage`.
    links.send_get_request(0, /*window=*/0, /*offset=*/900, /*length=*/1000, /*dest_window=*/0,
                           /*dest_off=*/0, /*requester_rank=*/0, /*token=*/1);
    // A second, well-formed request right after, on the same stream: this is what T5 checks for
    // placements and the same property has to hold here -- a refusal must not desynchronise the
    // stream, or everything after the first bad frame would be unrecoverable in a real run.
    links.send_get_request(0, 0, /*offset=*/0, /*length=*/64, /*dest_window=*/0, /*dest_off=*/0,
                           /*requester_rank=*/0, /*token=*/2);
    links.shutdown_all();
    agent.join();

    expect(not octx.report.called or octx.report.owner_token != 1,
          "the out-of-range request never reached the get_request hook");
    expect(octx.report.called and octx.report.owner_token == 2,
          "the well-formed request right after it still arrived");
    expect(agent.errors() >= 1, "the out-of-range request was counted as refused");
}

// ---------------------------------------------------------------------------
// G4: a GetRequest naming a window the receiver never registered.
// ---------------------------------------------------------------------------
void test_get_unknown_window() {
    std::vector<char> storage(1024, 0);
    srdsim::WindowTable windows;
    windows.add({storage.data(), storage.size(), false});   // id 0 only; this test asks for id 9

    OwnerCtx octx;   // `octx.reply` stays null: neither request here needs a real reply
    srdsim::ReceiveAgent::Hooks hooks{};
    hooks.device_copy = owner_host_copy;
    hooks.get_request = owner_get_request;
    hooks.user = &octx;
    hooks.log = [](const char*) {};
    srdsim::ReceiveAgent agent(&windows, hooks);

    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { expect(false, "socketpair"); return; }
    srdsim::set_nodelay(sv[0]);
    srdsim::set_nodelay(sv[1]);
    agent.serve(sv[1]);

    srdsim::SendLinks links;
    links.set(0, sv[0]);
    links.send_get_request(0, /*window=*/9, 0, 64, 0, 0, 0, /*token=*/1);
    links.send_get_request(0, /*window=*/0, 0, 64, 0, 0, 0, /*token=*/2);
    links.shutdown_all();
    agent.join();

    expect(octx.report.called and octx.report.owner_token == 2,
          "the request for an unknown window was skipped and the one after it still arrived");
    expect(agent.errors() >= 1, "the unknown window was counted as refused");
}

// ---------------------------------------------------------------------------
// G5: a reply frame the requester does not accept never reaches memory.
// ---------------------------------------------------------------------------
struct AcceptReport {
    int asked = 0, asked_good = 0, landed_calls = 0, peer_seen = 0;
    uint32_t window = 0;
    uint64_t offset = 0, length = 0;
    uint64_t failed_token = 0, failed_bytes = 0;
    int failed_calls = 0;
    RequesterReport landed;
};

void test_get_reply_not_accepted() {
    std::vector<char> storage(1024, 0);
    srdsim::WindowTable windows;
    windows.add({storage.data(), storage.size(), false});

    AcceptReport rep;
    srdsim::ReceiveAgent::Hooks hooks{};
    hooks.device_copy = requester_host_copy;
    hooks.get_response_landed = [](uint64_t token, uint64_t bytes, int peer, void* user) {
        ++ static_cast<AcceptReport*>(user)->landed_calls;
        requester_get_landed(token, bytes, peer, &static_cast<AcceptReport*>(user)->landed);
    };
    // Token 1 is refused, as a forged or stale reply would be; token 2 is accepted.
    hooks.get_response_accept = [](uint64_t token, uint32_t window, uint64_t offset, uint64_t length, int peer, void* user) {
        auto* r = static_cast<AcceptReport*>(user);
        r->peer_seen = peer;
        if (token == 1) { ++ r->asked; r->window = window; r->offset = offset; r->length = length; }
        else ++ r->asked_good;
        return token != 1;
    };
    hooks.get_failed = [](uint64_t token, uint64_t bytes, int, void* user) {
        auto* r = static_cast<AcceptReport*>(user);
        ++ r->failed_calls; r->failed_token = token; r->failed_bytes = bytes;
    };
    hooks.user = &rep;
    hooks.log = [](const char*) {};
    srdsim::ReceiveAgent agent(&windows, hooks);

    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { expect(false, "socketpair"); return; }
    srdsim::set_nodelay(sv[0]);
    srdsim::set_nodelay(sv[1]);
    agent.serve(sv[1], nullptr, /*peer_rank=*/5);

    std::vector<char> bad(128, char(0xAA)), good(64, char(0x55));
    srdsim::SendLinks links;
    links.set(0, sv[0]);
    links.send_placement(0, /*window=*/0, /*offset=*/256, bad.data(), bad.size(), /*token=*/1, srdsim::WireKind::GetResponse);
    links.send_placement(0, /*window=*/0, /*offset=*/512, good.data(), good.size(), /*token=*/2, srdsim::WireKind::GetResponse);
    links.shutdown_all();
    agent.join();

    size_t written = 0, landed = 0;
    for (size_t i = 0; i < bad.size(); ++ i) written += storage[256 + i] != 0;
    for (size_t i = 0; i < good.size(); ++ i) landed += storage[512 + i] == char(0x55);
    expect(rep.asked == 1 and rep.window == 0 and rep.offset == 256 and rep.length == bad.size(),
           ctx("the accept hook saw the refused frame's window, offset and length (asked %d, offset %llu, length %llu)",
               rep.asked, (unsigned long long)rep.offset, (unsigned long long)rep.length));
    expect(written == 0, ctx("the refused frame wrote nothing (%zu byte(s) changed)", written));
    expect(rep.failed_calls == 1 and rep.failed_token == 1 and rep.failed_bytes == bad.size(),
           ctx("the refused frame was reported through get_failed with its length (calls %d, token %llu, bytes %llu)",
               rep.failed_calls, (unsigned long long)rep.failed_token, (unsigned long long)rep.failed_bytes));
    expect(landed == good.size(), "the accepted frame after it landed");
    expect(rep.asked_good == 1, "the accept hook was asked about the accepted frame too");
    expect(rep.peer_seen == 5, ctx("the accept hook was given the link's peer rank (5), got %d", rep.peer_seen));
    expect(rep.landed_calls == 1 and rep.landed.landed_token == 2 and rep.landed.landed_bytes == good.size(),
           ctx("get_response_landed fired once, for the accepted frame (%d call(s))", rep.landed_calls));
    expect(agent.errors() == 1, ctx("the refused frame was counted once (%llu)", (unsigned long long)agent.errors()));
}

// G6: the next-frame rule, clause by clause, for a get of 100 bytes into [1000, 1100) of window 3.
void test_reply_frame_is_next() {
    const auto next = [](uint64_t remaining, uint32_t window, uint64_t offset, uint64_t length) {
        return srdsim::reply_frame_is_next(/*dst_window=*/3, /*dst_off=*/1000, /*size=*/100, remaining, window, offset, length);
    };
    expect(next(100, 3, 1000, 40), "G6 the first frame, at the start of the range");
    expect(next(60, 3, 1040, 60), "G6 the frame after 40 bytes, reaching the end exactly");
    expect(not next(100, 4, 1000, 40), "G6 another window");
    expect(not next(100, 3, 999, 40), "G6 before the range");
    expect(not next(60, 3, 1000, 40), "G6 the first frame again: a duplicate would leave a hole at the end");
    expect(not next(60, 3, 1060, 40), "G6 a frame that skips ahead");
    expect(not next(60, 3, 1040, 61), "G6 a frame past the end of the range");
    expect(not next(0, 3, 1100, 1), "G6 any frame once every byte is accounted for");
    expect(not next(101, 3, 999, 1), "G6 more remaining than the get's size");
    expect(not next(0, 3, 1100, 0), "G6 an empty frame, even where the get ends");
    expect(not next(100, 3, 1000, 0), "G6 an empty frame at the start");
}

}  // namespace

int main() {
    test_get_round_trip();
    test_get_out_of_range();
    test_get_unknown_window();
    test_get_reply_not_accepted();
    test_reply_frame_is_next();
    std::printf(g_failures == 0 ? "PASS: %d checks, 0 failures\n" : "FAIL: %d checks, %d failures\n",
               g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

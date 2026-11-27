// SPDX-License-Identifier: Apache-2.0
// Abuses an endpoint's listening port with malformed and hostile frames, then checks that it still serves a real peer.
//
//   test_fi_wire PROVIDER_SO [ROUNDS]
//
// The provider listens on the loopback interface without authentication, so any process on the host can send it
// frames. This test connects to an endpoint's port as such a process: frames announcing a size they do not carry,
// pieces of a message that claim more than the message announced, writes and reads that straddle or overflow the
// registered region, acknowledgements of nothing, and streams of random frames seeded from ROUNDS. After every
// connection a second endpoint sends a message to the first and a write into its region, and both must arrive.
// A crash fails the test; built with a sanitizer, so does a memory error.

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// The provider's frame, as it is on the wire (src/fi/srdsim_xfer.c).
struct Frame {
    uint32_t type, flags;
    uint64_t op_id, offset, len, total, addr, key, data, tag;
};
static_assert(sizeof(Frame) == 72, "the wire frame changed: update this mirror and the provider together");
enum { kHello = 1, kMsg, kWrite, kReadReq, kReadResp, kAck };

int g_failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (not ok) ++g_failures;
}
[[noreturn]] void die(const char* what, int rc) {
    std::printf("FAIL %s: %d (%s)\n", what, rc, fi_strerror(-rc));
    std::fflush(stdout);
    std::_Exit(1);
}

struct Node {
    fi_info* info = nullptr;
    fid_fabric* fabric = nullptr;
    fid_domain* domain = nullptr;
    fid_av* av = nullptr;
    fid_cq* cq = nullptr;
    fid_ep* ep = nullptr;
    char name[64] = {};
    fi_addr_t peer = FI_ADDR_NOTAVAIL;
    std::vector<uint8_t> region;
    fid_mr* mr = nullptr;
    uint64_t key = 0;
};

void open_node(Node* n) {
    fi_info* hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->domain_attr->threading = FI_THREAD_COMPLETION;
    hints->fabric_attr->prov_name = strdup("efa");
    hints->tx_attr->msg_order = FI_ORDER_SAS;
    hints->rx_attr->msg_order = FI_ORDER_SAS;
    int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &n->info);
    fi_freeinfo(hints);
    if (rc) die("fi_getinfo", rc);
    if ((rc = fi_fabric(n->info->fabric_attr, &n->fabric, nullptr))) die("fi_fabric", rc);
    if ((rc = fi_domain(n->fabric, n->info, &n->domain, nullptr))) die("fi_domain", rc);
    fi_av_attr av_attr = {};
    av_attr.type = FI_AV_TABLE;
    if ((rc = fi_av_open(n->domain, &av_attr, &n->av, nullptr))) die("fi_av_open", rc);
    fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_TAGGED;
    if ((rc = fi_cq_open(n->domain, &cq_attr, &n->cq, nullptr))) die("fi_cq_open", rc);
    if ((rc = fi_endpoint(n->domain, n->info, &n->ep, nullptr))) die("fi_endpoint", rc);
    if ((rc = fi_ep_bind(n->ep, &n->av->fid, 0))) die("bind av", rc);
    if ((rc = fi_ep_bind(n->ep, &n->cq->fid, FI_TRANSMIT | FI_RECV))) die("bind cq", rc);
    if ((rc = fi_enable(n->ep))) die("fi_enable", rc);
    size_t len = sizeof n->name;
    if ((rc = fi_getname(&n->ep->fid, n->name, &len))) die("fi_getname", rc);
    n->region.assign(64 << 10, 0);
    rc = fi_mr_reg(n->domain, n->region.data(), n->region.size(), FI_REMOTE_WRITE | FI_REMOTE_READ, 0, 0, 0, &n->mr, nullptr);
    if (rc) die("fi_mr_reg", rc);
    n->key = fi_mr_key(n->mr);
}

uint64_t g_rng = 0x9e3779b97f4a7c15ull;
uint64_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

// Values that sit at the edges of sizes, offsets and counters, with the registered region's own address and key mixed in.
uint64_t edge(const Node& victim) {
    const uint64_t base = uint64_t(uintptr_t(victim.region.data()));
    static const uint64_t fixed[] = {0, 1, 7, 8, 31, 32, 33, 4095, 4096, 8192, 8193, 65535, 65536, 1u << 20, (1ull << 32) - 1,
                                     1ull << 32, 1ull << 40, 1ull << 62, 1ull << 63, ~0ull - 1, ~0ull};
    switch (rnd() % 6) {
        case 0: return base + (rnd() % 70000);
        case 1: return base - (rnd() % 64);
        case 2: return victim.key;
        case 3: return victim.key ^ (1ull << (rnd() % 32));
        default: return fixed[rnd() % (sizeof fixed / sizeof fixed[0])];
    }
}

int dial(const char* name) {
    uint16_t port;
    uint32_t ipv4;
    std::memcpy(&port, name + 12, sizeof port);
    std::memcpy(&ipv4, name + 24, sizeof ipv4);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) die("socket", -errno);
    sockaddr_in sin = {};
    sin.sin_family = AF_INET;
    sin.sin_port = port;
    sin.sin_addr.s_addr = ipv4;
    if (connect(fd, reinterpret_cast<sockaddr*>(&sin), sizeof sin) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

void send_all(int fd, const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    while (n) {
        ssize_t w = send(fd, c, n, MSG_NOSIGNAL);
        if (w <= 0) return;   // the provider closed the connection, which is an answer to this frame
        c += w;
        n -= size_t(w);
    }
}

void send_frame(int fd, const Frame& f, const void* payload, size_t payload_len) {
    send_all(fd, &f, sizeof f);
    if (payload_len) send_all(fd, payload, payload_len);
}

void hello(int fd, const char* victim_name) {
    char me[32];
    std::memcpy(me, victim_name, sizeof me);       // a well-formed address of an endpoint that does not exist
    me[16] ^= 0x33;
    Frame f = {};
    f.type = kHello;
    f.len = sizeof me;
    send_frame(fd, f, me, sizeof me);
}

// Returns once the provider has had time to read what was sent.
void settle() { usleep(30000); }

void normal_traffic(Node* victim, Node* peer, const char* what) {
    static int round = 0;
    ++round;
    uint8_t in[64] = {};
    if (int rc = int(fi_recv(victim->ep, in, sizeof in, nullptr, FI_ADDR_UNSPEC, nullptr))) die("fi_recv", rc);
    uint8_t out[64];
    std::memset(out, 0x30 + round % 40, sizeof out);
    if (int rc = int(fi_send(peer->ep, out, sizeof out, nullptr, peer->peer, nullptr))) die("fi_send", rc);
    std::vector<uint8_t> w(4096, uint8_t(0x50 + round % 40));
    const uint64_t at = uint64_t(uintptr_t(victim->region.data())) + 8192;
    if (int rc = int(fi_write(peer->ep, w.data(), w.size(), nullptr, peer->peer, at, victim->key, nullptr))) die("fi_write", rc);
    int sent = 0, got = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while ((sent < 2 or got < 1) and std::chrono::steady_clock::now() < deadline) {
        fi_cq_tagged_entry c;
        ssize_t r = fi_cq_read(peer->cq, &c, 1);
        if (r == 1) ++sent;
        else if (r != -FI_EAGAIN) die("peer cq", int(r));
        r = fi_cq_read(victim->cq, &c, 1);
        if (r == 1) ++got;
        else if (r != -FI_EAGAIN and r != -FI_EAVAIL) die("victim cq", int(r));
        else if (r == -FI_EAVAIL) { fi_cq_err_entry e = {}; fi_cq_readerr(victim->cq, &e, 0); }
    }
    bool ok = sent == 2 and got == 1 and std::memcmp(in, out, sizeof in) == 0;
    for (size_t k = 0; k < w.size() and ok; ++k) ok = victim->region[8192 + k] == w[k];
    char line[160];
    std::snprintf(line, sizeof line, "after %s, a real peer's send and write still arrive", what);
    check(ok, line);
}

void crafted(Node* victim, Node* peer) {
    const uint64_t base = uint64_t(uintptr_t(victim->region.data()));
    std::vector<uint8_t> big(8192, 0xab);

    {   // A piece that claims more than its message announced: the heap overflow found in review.
        int fd = dial(victim->name);
        hello(fd, victim->name);
        Frame a = {};
        a.type = kMsg; a.op_id = 1; a.offset = 0; a.len = 8; a.total = 16;
        send_frame(fd, a, big.data(), 8);
        Frame b = a;
        b.offset = 8; b.len = 4096; b.total = 1u << 20;
        send_frame(fd, b, big.data(), 4096);
        settle();
        close(fd);
        normal_traffic(victim, peer, "a piece that claims more than its message announced");
    }
    {   // A header that announces a payload of a terabyte, then silence.
        int fd = dial(victim->name);
        hello(fd, victim->name);
        Frame f = {};
        f.type = kMsg; f.op_id = 2; f.len = 1ull << 40; f.total = 1ull << 40;
        send_frame(fd, f, nullptr, 0);
        settle();
        close(fd);
        normal_traffic(victim, peer, "a header announcing a terabyte");
    }
    {   // Writes that straddle and overflow the registered region.
        int fd = dial(victim->name);
        hello(fd, victim->name);
        const uint64_t at[] = {base + victim->region.size() - 100, base - 50, ~0ull - 20, base + 10};
        for (uint64_t a : at) {
            Frame f = {};
            f.type = kWrite; f.op_id = 3; f.offset = 0; f.len = 4096; f.total = 4096; f.addr = a; f.key = victim->key;
            send_frame(fd, f, big.data(), 4096);
        }
        Frame f = {};
        f.type = kWrite; f.op_id = 4; f.offset = ~0ull - 100; f.len = 200; f.total = ~0ull; f.addr = base; f.key = victim->key;
        send_frame(fd, f, big.data(), 200);
        settle();
        close(fd);
        normal_traffic(victim, peer, "writes straddling and overflowing the region");
    }
    {   // Reads of more than the region holds, and of an address range that wraps.
        int fd = dial(victim->name);
        hello(fd, victim->name);
        const uint64_t at[] = {base, base + victim->region.size() - 8, ~0ull - 7};
        const uint64_t total[] = {1ull << 30, 4096, 4096};
        for (int i = 0; i < 3; ++i) {
            Frame f = {};
            f.type = kReadReq; f.op_id = 5 + uint64_t(i); f.total = total[i]; f.addr = at[i]; f.key = victim->key;
            send_frame(fd, f, nullptr, 0);
        }
        settle();
        close(fd);
        normal_traffic(victim, peer, "reads beyond the region and across the address space");
    }
    {   // Acknowledgements and read responses of nothing, with and without payload.
        int fd = dial(victim->name);
        hello(fd, victim->name);
        for (uint64_t id = 1; id < 40; id += 3) {
            Frame f = {};
            f.type = kAck; f.op_id = id; f.data = ~0ull;
            send_frame(fd, f, nullptr, 0);
            Frame r = {};
            r.type = kReadResp; r.op_id = id; r.offset = 0; r.len = 64; r.total = 64;
            send_frame(fd, r, big.data(), 64);
        }
        settle();
        close(fd);
        normal_traffic(victim, peer, "acknowledgements and read responses of nothing");
    }
    {   // A frame cut off in the middle of its header, and in the middle of its payload.
        int fd = dial(victim->name);
        hello(fd, victim->name);
        Frame f = {};
        f.type = kMsg; f.op_id = 9; f.len = 4096; f.total = 4096;
        send_all(fd, &f, sizeof f - 5);
        close(fd);
        fd = dial(victim->name);
        hello(fd, victim->name);
        send_all(fd, &f, sizeof f);
        send_all(fd, big.data(), 1000);
        close(fd);
        settle();
        normal_traffic(victim, peer, "frames cut off mid-header and mid-payload");
    }
}

void random_streams(Node* victim, Node* peer, int rounds) {
    std::vector<uint8_t> noise(1 << 17);
    for (int r = 0; r < rounds; ++r) {
        int fd = dial(victim->name);
        if (fd < 0) { check(false, "the endpoint still accepts connections"); return; }
        if (rnd() % 4) hello(fd, victim->name);
        const int frames = 1 + int(rnd() % 12);
        for (int i = 0; i < frames; ++i) {
            Frame f = {};
            f.type = (rnd() % 9 == 0) ? uint32_t(rnd()) : uint32_t(1 + rnd() % 6);
            f.flags = (rnd() % 5 == 0) ? uint32_t(rnd()) : uint32_t(rnd() % 8);
            f.op_id = rnd() % 6;
            f.offset = edge(*victim);
            f.len = (rnd() % 3) ? rnd() % 5000 : edge(*victim);
            f.total = edge(*victim);
            f.addr = edge(*victim);
            f.key = edge(*victim);
            f.data = edge(*victim);
            f.tag = rnd();
            for (auto& b : noise) b = uint8_t(rnd());
            const size_t have = f.len <= noise.size() ? size_t(f.len) : 0;
            send_frame(fd, f, noise.data(), have);
            if (f.len > noise.size()) break;   // the rest of the stream would only be read as that payload
        }
        settle();
        close(fd);
        if (r % 10 == 9 or r == rounds - 1) normal_traffic(victim, peer, "a stream of random frames");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s PROVIDER_SO [ROUNDS]\n", argv[0]); return 2; }
    setenv("FI_PROVIDER_PATH", (std::string("+") + argv[1]).c_str(), 1);
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 100;
    static Node victim, peer;
    open_node(&victim);
    open_node(&peer);
    if (fi_av_insert(victim.av, peer.name, 1, &victim.peer, 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);
    if (fi_av_insert(peer.av, victim.name, 1, &peer.peer, 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);
    normal_traffic(&victim, &peer, "nothing");
    crafted(&victim, &peer);
    random_streams(&victim, &peer, rounds);
    std::printf("wire: %s\n", g_failures == 0 ? "passed" : "failed");
    // _Exit does not flush stdio, and with stdout redirected to a file that silently threw away
    // everything still in the buffer -- including this verdict line, so a log that looked complete
    // ended on a PASS and said nothing about the result.
    std::fflush(nullptr);
    std::_Exit(g_failures == 0 ? 0 : 1);   // closing the endpoints with hostile connections open is not what is tested here
}

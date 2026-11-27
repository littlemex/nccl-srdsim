// SPDX-License-Identifier: Apache-2.0
// On a host that has a real EFA device: naming this provider replaces the device completely, so no
// request can land on the real NIC while a test believes it is using this provider.
//
//   test_fi_beside_efa PROVIDER_SO
//
// NEEDS A REAL EFA DEVICE. CMake does not register this as a test and CI does not run it; it is built
// with -DSRDSIM_BUILD_FI_EFA_TEST=ON and run by hand on an EFA instance, as the GPU tests are.
//
// Why this is the property worth checking. This provider registers under the name `efa` so that a
// consumer's EFA path is taken without changing the consumer. On a host that also has the device,
// that raises a question with a bad answer and a good one: the bad answer would be that some requests
// reach this provider and others reach the NIC, which would make an experiment quietly meaningless --
// an ordering this provider was asked to produce would not be produced, and traffic a test meant to
// keep inside one host would leave it. The good answer, which is what libfabric does, is that the
// preferred provider takes the name whole: with this provider named, `fi_getinfo` for `efa` returns
// this provider's domains and none of the device's, and with it not named, the device's and none of
// this provider's. Both are checked here, in one process, against the device that is actually present.
//
// The consequence for anyone using this: the device and this provider cannot be used at the same time
// through the name `efa`. A job uses one or the other. Two ranks cannot split across them anyway --
// this is a separate fabric with its own address format, as the documents say -- and this is the
// mechanism that makes it impossible to do by accident.

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kMessages = 50;
constexpr size_t kLen = 4096;

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (not ok) ++g_failures;
}

[[noreturn]] void die(const char* what, int rc) {
    std::printf("FAIL %s: %d (%s)\n", what, rc, fi_strerror(-rc));
    std::fflush(stdout);
    std::fflush(nullptr);
    std::_Exit(1);
}

bool is_ours(const fi_info* i) {
    return i->domain_attr and i->domain_attr->name and std::strstr(i->domain_attr->name, "srdsim");
}

// Every `efa` result in this process, as a count of ours and of anything else, with the first domain
// name of each kind for the report.
struct Census {
    int ours = 0, other = 0;
    std::string first_ours, first_other;
};

Census census() {
    fi_info* hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->fabric_attr->prov_name = strdup("efa");
    fi_info* list = nullptr;
    const int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &list);
    fi_freeinfo(hints);
    Census c;
    if (rc == -FI_ENODATA) return c;
    if (rc) die("fi_getinfo for efa", rc);
    for (const fi_info* i = list; i; i = i->next) {
        const char* d = i->domain_attr and i->domain_attr->name ? i->domain_attr->name : "(none)";
        if (is_ours(i)) { if (!c.ours++) c.first_ours = d; }
        else { if (!c.other++) c.first_other = d; }
    }
    fi_freeinfo(list);
    return c;
}

// One endpoint pair on the first `efa` result, and kMessages carried between them. Proves that what
// answered to the name on this host actually works, rather than only appearing in a listing.
int carry_one_pair() {
    fi_info* hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->fabric_attr->prov_name = strdup("efa");
    fi_info* info = nullptr;
    int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &info);
    fi_freeinfo(hints);
    if (rc) die("fi_getinfo", rc);

    fid_fabric* fabric = nullptr;
    fid_domain* domain = nullptr;
    if ((rc = fi_fabric(info->fabric_attr, &fabric, nullptr))) die("fi_fabric", rc);
    if ((rc = fi_domain(fabric, info, &domain, nullptr))) die("fi_domain", rc);

    struct Side {
        fid_av* av = nullptr;
        fid_cq* cq = nullptr;
        fid_ep* ep = nullptr;
        char name[64] = {};
        fi_addr_t peer = 0;
    } side[2];
    for (Side& s : side) {
        fi_av_attr av_attr = {};
        av_attr.type = FI_AV_TABLE;
        if ((rc = fi_av_open(domain, &av_attr, &s.av, nullptr))) die("fi_av_open", rc);
        fi_cq_attr cq_attr = {};
        cq_attr.format = FI_CQ_FORMAT_TAGGED;
        if ((rc = fi_cq_open(domain, &cq_attr, &s.cq, nullptr))) die("fi_cq_open", rc);
        if ((rc = fi_endpoint(domain, info, &s.ep, nullptr))) die("fi_endpoint", rc);
        if ((rc = fi_ep_bind(s.ep, &s.av->fid, 0))) die("bind av", rc);
        if ((rc = fi_ep_bind(s.ep, &s.cq->fid, FI_TRANSMIT | FI_RECV))) die("bind cq", rc);
        if ((rc = fi_enable(s.ep))) die("fi_enable", rc);
        size_t len = sizeof s.name;
        if ((rc = fi_getname(&s.ep->fid, s.name, &len))) die("fi_getname", rc);
    }
    if (fi_av_insert(side[0].av, side[1].name, 1, &side[0].peer, 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);
    if (fi_av_insert(side[1].av, side[0].name, 1, &side[1].peer, 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);

    std::vector<std::vector<uint8_t>> out(kMessages), in(kMessages);
    for (int i = 0; i < kMessages; ++i) {
        out[i].assign(kLen, 0);
        in[i].assign(kLen, 0);
        for (size_t k = 0; k < kLen; ++k) out[i][k] = uint8_t((i * 31 + k * 7) & 0xff);
        while (fi_recv(side[1].ep, in[i].data(), kLen, nullptr, FI_ADDR_UNSPEC, &in[i]) == -FI_EAGAIN)
            fi_cq_read(side[1].cq, nullptr, 0);
    }
    int sent = 0, got = 0, wrong = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while ((sent < kMessages or got < kMessages) and std::chrono::steady_clock::now() < deadline) {
        if (sent < kMessages) {
            const ssize_t s = fi_send(side[0].ep, out[sent].data(), kLen, nullptr, side[0].peer, &out[sent]);
            if (s == 0) ++sent;
            else if (s != -FI_EAGAIN) die("fi_send", int(s));
        }
        fi_cq_tagged_entry e[8];
        fi_cq_read(side[0].cq, e, 8);
        const ssize_t n = fi_cq_read(side[1].cq, e, 8);
        if (n > 0) got += int(n);
    }
    if (sent != kMessages or got != kMessages) {
        std::printf("sent %d of %d, received %d of %d\n", sent, kMessages, got, kMessages);
        ++wrong;
    }
    for (int i = 0; i < kMessages; ++i)
        for (size_t k = 0; k < kLen; ++k)
            if (in[i][k] != uint8_t((i * 31 + k * 7) & 0xff)) { ++wrong; k = kLen; }

    for (Side& s : side) {
        if (s.ep) fi_close(&s.ep->fid);
        if (s.cq) fi_close(&s.cq->fid);
        if (s.av) fi_close(&s.av->fid);
    }
    fi_close(&domain->fid);
    fi_close(&fabric->fid);
    fi_freeinfo(info);
    return wrong;
}

}  // namespace

// libfabric reads FI_PROVIDER_PATH once, when it initialises on the first fi_getinfo, so the two
// halves cannot be done in one process: setting the variable after that call would have no effect and
// the test would report the device as absent for the wrong reason. The driver runs itself twice.
int main(int argc, char** argv) {
    if (argc == 3 and std::strcmp(argv[2], "device") == 0) {
        unsetenv("FI_PROVIDER_PATH");
        Census c = census();
        std::printf("not named: %d results from the device (%s), %d from this provider\n",
                    c.other, c.first_other.empty() ? "-" : c.first_other.c_str(), c.ours);
        check(c.other > 0, "a real EFA device is present and answers to the name efa");
        check(c.ours == 0, "this provider does not answer before it is named");
        if (c.other > 0) check(carry_one_pair() == 0, "the device carries messages between two endpoints on this host");
        std::fflush(nullptr);
        return g_failures == 0 ? 0 : 1;
    }
    if (argc == 3 and std::strcmp(argv[2], "named") == 0) {
        setenv("FI_PROVIDER_PATH", (std::string("+") + argv[1]).c_str(), 1);
        Census c = census();
        std::printf("named: %d results from this provider (%s), %d from the device (%s)\n",
                    c.ours, c.first_ours.empty() ? "-" : c.first_ours.c_str(),
                    c.other, c.first_other.empty() ? "-" : c.first_other.c_str());
        check(c.ours > 0, "this provider answers to the name efa once it is named");
        check(c.other == 0, "the device no longer answers to the name efa, so no request can reach it by accident");
        if (c.ours > 0) check(carry_one_pair() == 0, "this provider carries messages between two endpoints on the same host");
        std::fflush(nullptr);
        return g_failures == 0 ? 0 : 1;
    }
    if (argc != 2) { std::fprintf(stderr, "usage: %s PROVIDER_SO\n", argv[0]); return 2; }

    int bad = 0;
    for (const char* mode : {"device", "named"}) {
        std::fflush(nullptr);
        const pid_t pid = fork();
        if (pid < 0) { std::perror("fork"); return 2; }
        if (pid == 0) {
            execl("/proc/self/exe", argv[0], argv[1], mode, static_cast<char*>(nullptr));
            execl(argv[0], argv[0], argv[1], mode, static_cast<char*>(nullptr));
            std::perror("exec");
            std::_Exit(2);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        if (not WIFEXITED(st) or WEXITSTATUS(st) != 0) {
            std::printf("the %s pass failed (status %d)\n", mode, st);
            ++bad;
        }
    }
    std::printf("beside_efa: %s\n", bad == 0 ? "passed" : "failed");
    std::fflush(nullptr);
    return bad == 0 ? 0 : 1;
}

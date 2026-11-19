// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Deliberate impairments: delay, burstiness, backpressure and a frozen link, applied on purpose.
//
// WHY THIS EXISTS, STATED PLAINLY
// Everything else in this directory checks **safety**: no reachable state releases a signal before its
// payload, no frame sequence leaves the receiver reading a header from inside a payload, no input makes
// the packing ambiguous. Those are properties of states, and enumerating states or handing them to a
// solver settles them.
//
// Every defect that actually cost time here was a **liveness** failure instead:
//
//   * a progress callback that stopped returning, because it did a blocking write
//   * a device copy that waited for a kernel that was waiting for the copy
//   * a send queue that grew and never drained
//   * a monitor that could not report, because its subject had blocked it
//   * a test that deadlocked on a machine slower than the one it was written on
//
// Not one of those is visible to a state enumeration. "Something good eventually happens" is not a
// property of any single state, and a model that abstracts time away cannot see a thread that is not
// making progress. They were all found by running on rented hardware and staring at the symptom.
//
// So the fabric is given knobs to be slow, bursty, congested and one-sidedly frozen, and the tests
// assert bounds on time rather than facts about states. Two things follow. The harness can be shown to
// *detect* a liveness failure rather than hoped to, by injecting one and requiring the report. And the
// code under test -- DeepEP's kernels -- can be run against a fabric that behaves badly on purpose,
// which is the situation a real one will eventually produce and which no amount of state enumeration
// will ever reach.
//
// CALIBRATION
// The same rule as the scheduler's: do what a real fabric may do, never what it may not. SRD is
// reliable, so nothing here drops or duplicates. What it does do is take its time, take it unevenly,
// and take longer on one path than another -- all of which EFA does.
//
// Everything is off by default. An impairment that is on unless disabled turns every unrelated failure
// into a hunt for which knob was set.

#ifndef SRDSIM_IMPAIR_H_
#define SRDSIM_IMPAIR_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>

namespace srdsim {

// One knob per way a fabric can be slow. Read from the environment so a run can be impaired without
// rebuilding, and settable directly so a test can be explicit.
struct Impairments {
    // Every released unit waits this long before it is handed to the wire. A fixed floor with a random
    // span, because a constant delay is the one shape that hides ordering problems: it preserves
    // relative order exactly.
    uint32_t delay_us_min = 0;
    uint32_t delay_us_max = 0;

    // Release in bursts: `burst` units back to back, then a pause. Real fabrics are bursty and a
    // steady trickle is the easiest case for a receiver that batches.
    uint32_t burst = 0;
    uint32_t burst_pause_us = 0;

    // The receiving side applies at most one frame every this many microseconds. This is the shape that
    // produces backpressure without any failure anywhere: a receiver that is merely slow.
    uint32_t apply_interval_us = 0;

    // Shrink the socket send buffer, which is the cheap way to force a partial send. A transport that
    // assumes a whole frame fits is wrong on a congested link and right on an idle one, so the small
    // buffer is how the wrong version gets caught.
    uint32_t send_buffer_bytes = 0;

    // Freeze one peer's link: accept nothing from it for this long, starting after this many units have
    // been applied. One frozen path while the others run is the case that separates "the fabric is
    // slow" from "this simulator has a global bottleneck".
    int      freeze_peer = -1;
    uint32_t freeze_after_units = 0;
    uint32_t freeze_ms = 0;

    // Hold a signal behind its own payload by this much extra. The weak bundled-signal contract says a
    // signal is observable only after its payload lands; this widens the gap so a reader that peeks
    // early is caught, without ever breaking the contract.
    uint32_t signal_lag_us = 0;

    bool any() const {
        return delay_us_max != 0 or burst != 0 or apply_interval_us != 0 or
               send_buffer_bytes != 0 or (freeze_peer >= 0 and freeze_ms != 0) or signal_lag_us != 0;
    }

    std::string describe() const {
        if (not any()) return "none";
        std::string s;
        const auto add = [&s](const std::string& part) {
            if (not s.empty()) s += ", ";
            s += part;
        };
        if (delay_us_max) add("delay " + std::to_string(delay_us_min) + ".." +
                              std::to_string(delay_us_max) + "us");
        if (burst) add("bursts of " + std::to_string(burst) + " then " +
                       std::to_string(burst_pause_us) + "us");
        if (apply_interval_us) add("apply every " + std::to_string(apply_interval_us) + "us");
        if (send_buffer_bytes) add("send buffer " + std::to_string(send_buffer_bytes) + "B");
        if (freeze_peer >= 0 and freeze_ms)
            add("freeze peer " + std::to_string(freeze_peer) + " for " + std::to_string(freeze_ms) +
                "ms after " + std::to_string(freeze_after_units) + " units");
        if (signal_lag_us) add("signal lag " + std::to_string(signal_lag_us) + "us");
        return s;
    }
};

inline uint32_t impair_env(const char* name, uint32_t dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr or *v == '\0') return dflt;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(v, &end, 10);
    return (end != nullptr and *end == '\0') ? static_cast<uint32_t>(parsed) : dflt;
}

// The names are spelled out rather than packed into one variable, so a run's impairment shows up in
// its environment one line at a time and can be changed one at a time.
inline Impairments impairments_from_env() {
    Impairments i;
    i.delay_us_min = impair_env("SRDSIM_DELAY_US_MIN", 0);
    i.delay_us_max = impair_env("SRDSIM_DELAY_US_MAX", 0);
    if (i.delay_us_max < i.delay_us_min) i.delay_us_max = i.delay_us_min;
    i.burst = impair_env("SRDSIM_BURST", 0);
    i.burst_pause_us = impair_env("SRDSIM_BURST_PAUSE_US", 0);
    i.apply_interval_us = impair_env("SRDSIM_APPLY_INTERVAL_US", 0);
    i.send_buffer_bytes = impair_env("SRDSIM_SEND_BUFFER_BYTES", 0);
    const uint32_t peer = impair_env("SRDSIM_FREEZE_PEER", 0xffffffffu);
    i.freeze_peer = (peer == 0xffffffffu) ? -1 : static_cast<int>(peer);
    i.freeze_after_units = impair_env("SRDSIM_FREEZE_AFTER_UNITS", 0);
    i.freeze_ms = impair_env("SRDSIM_FREEZE_MS", 0);
    i.signal_lag_us = impair_env("SRDSIM_SIGNAL_LAG_US", 0);
    return i;
}

// Sleeps the way an impairment asks, using the same PRNG discipline as the scheduler: a seed in, a
// reproducible sequence out. A random delay nobody can replay turns a counterexample into an anecdote.
class Delayer {
public:
    explicit Delayer(const Impairments& imp, uint64_t seed)
        : imp_(imp), state_(seed ? seed : 0x9E3779B97F4A7C15ull) {}

    // Called once per released unit. Returns immediately when nothing is configured, which is the
    // default, so an unimpaired run pays one comparison.
    void before_unit(bool is_signal) {
        if (imp_.burst != 0) {
            if (++ in_burst_ >= imp_.burst) {
                in_burst_ = 0;
                sleep_us(imp_.burst_pause_us);
            }
        }
        if (imp_.delay_us_max != 0) {
            const uint32_t span = imp_.delay_us_max - imp_.delay_us_min;
            sleep_us(imp_.delay_us_min + (span ? static_cast<uint32_t>(next() % (span + 1)) : 0));
        }
        if (is_signal and imp_.signal_lag_us != 0)
            sleep_us(imp_.signal_lag_us);
    }

private:
    uint64_t next() {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * 0x2545F4914F6CDD1Dull;
    }

    static void sleep_us(uint32_t us) {
        if (us == 0) return;
        std::this_thread::sleep_for(std::chrono::microseconds(us));
    }

    Impairments imp_;
    uint64_t state_;
    uint32_t in_burst_ = 0;
};

// A link that can be frozen on purpose, after a chosen amount of traffic. Shared by the receiving
// agent, which is where "accept nothing for a while" is expressible without lying about the wire.
class Freezer {
public:
    void configure(const Impairments& imp) { imp_ = imp; }

    // Called by the receiving side before applying a frame from `peer`. Returns after the freeze, if
    // one applies. `applied` is the number of units this rank has applied so far.
    void maybe_freeze(int peer, uint64_t applied) {
        if (imp_.freeze_peer < 0 or imp_.freeze_ms == 0) return;
        if (peer != imp_.freeze_peer) return;
        if (applied < imp_.freeze_after_units) return;
        bool expected = false;
        if (not frozen_once_.compare_exchange_strong(expected, true)) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(imp_.freeze_ms));
    }

    bool has_frozen() const { return frozen_once_.load(std::memory_order_relaxed); }

private:
    Impairments imp_;
    std::atomic<bool> frozen_once_{false};
};

}   // namespace srdsim

#endif   // SRDSIM_IMPAIR_H_

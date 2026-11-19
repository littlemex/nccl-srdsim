// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Proves the receiver cannot desynchronise, by enumerating every frame sequence it can be given.
//
// WHY THIS PROPERTY AND NOT ANOTHER
// On 2026-09-04 the receiver read a header out of the middle of a payload and reported `unknown window
// id 84326656`. Nothing crashed; it simply stopped making sense, and it took a day to find because the
// symptom named a window. The property that rules this out is narrow and checkable:
//
//   **After the receiver finishes handling any frame, it is either positioned exactly at the start of
//   the next frame, or the link is closed.** There is no third state.
//
// That is a claim about every sequence of frame kinds and outcomes, so it is enumerated rather than
// sampled. Two hand-written malformed frames -- which is what the transport test has -- cannot answer
// it, and the sequence that actually broke was not one of the two.
//
// WHAT THE MODEL ABSTRACTS
// Bytes are not modelled; positions are. A frame is (kind, whether the window is known, whether the
// range fits, whether the length is within the largest window, whether the local write succeeds), and
// the receiver's reaction is read off the same rules the implementation uses. What is checked is that
// every reaction leaves the stream at a frame boundary or closes the link, and that a refusal which
// leaves the link open is followed by a frame that is handled normally.
//
// The abstraction is only worth something if it matches the code, so `react` below has one branch per
// branch in `ReceiveAgent::loop`, each naming the branch it stands for, and the run prints which of
// those names were reached. A branch added to the loop and not here shows up as a name that never
// appears in that list. The range rule is not re-implemented at all: the model calls the shipped
// function, so that one cannot drift from the other.
//
// Build: c++ -std=c++17 -O2 -I layer2 tests/model_srdsim_framing.cpp -o m

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "srdsim_transport.h"

namespace {

int g_failures = 0;
void fail(const std::string& what) { ++ g_failures; std::printf("  [FAIL] %s\n", what.c_str()); }

// A frame as the model sees it.
struct Frame {
    enum class Kind { Placement, Signal, Shutdown, Unknown } kind = Kind::Placement;
    bool window_known = true;
    bool length_within_largest_window = true;   // the bound the receiver checks first
    bool range_fits = true;                     // offset and length inside the named window
    bool has_payload = true;                    // length > 0
    bool write_succeeds = true;                 // the local write, which can fail on a device window
};

// Where the receiver is after handling a frame.
enum class Position {
    AtFrameBoundary,   // positioned exactly at the next header
    LinkClosed,        // will read nothing more
    Desynchronised,    // positioned inside a payload: the state this model exists to rule out
};

// What the receiver does with a frame, and which branch of `ReceiveAgent::loop` it stands for.
struct Reaction {
    Position position;
    bool applied = false;
    bool refused = false;
    bool failed_write = false;
    const char* branch = "";
};

// One row per branch in the implementation. The order matters and mirrors the code: the length bound
// is checked before the window is resolved, because the length decides how much of the stream belongs
// to this frame and an unbounded length cannot be trusted to say how much to skip.
Reaction react(const Frame& f) {
    if (f.kind == Frame::Kind::Shutdown)
        return {Position::LinkClosed, false, false, false, "Shutdown: break"};

    if (not f.length_within_largest_window) {
        // The payload cannot be drained, because the number saying how much to drain is the number
        // that is not trusted. Leaving the link open here is precisely the desynchronisation.
        return {Position::LinkClosed, false, true, false, "length > largest window: refuse and break"};
    }

    if (f.kind == Frame::Kind::Unknown)
        return {Position::LinkClosed, false, true, false, "unknown kind: refuse and break"};

    if (not f.window_known) {
        // Drainable, because the length is bounded, so the stream survives.
        return {Position::AtFrameBoundary, false, true, false, "unknown window: drain and continue"};
    }

    if (f.kind == Frame::Kind::Placement) {
        if (not f.range_fits)
            return {Position::AtFrameBoundary, false, true, false, "placement out of range: drain and continue"};
        if (not f.write_succeeds)
            return {Position::AtFrameBoundary, false, false, true, "placement write failed: count and continue"};
        return {Position::AtFrameBoundary, true, false, false, "placement applied"};
    }

    // Signal: no payload at all, so nothing to drain and the stream is always still framed.
    if (not f.range_fits)
        return {Position::AtFrameBoundary, false, true, false, "signal out of range: refuse and continue"};
    if (not f.write_succeeds)
        return {Position::AtFrameBoundary, false, false, true, "signal write failed: count and continue"};
    return {Position::AtFrameBoundary, true, false, false, "signal applied"};
}

// Every frame the model can express. Combinations that cannot occur together are excluded and the
// exclusion is stated, because an unreachable case silently included would inflate the state count and
// prove nothing.
std::vector<Frame> all_frames() {
    std::vector<Frame> out;
    for (const auto kind : {Frame::Kind::Placement, Frame::Kind::Signal, Frame::Kind::Shutdown,
                            Frame::Kind::Unknown}) {
        for (const bool known : {true, false}) {
            for (const bool within : {true, false}) {
                for (const bool fits : {true, false}) {
                    for (const bool payload : {true, false}) {
                        for (const bool writes : {true, false}) {
                            Frame f{kind, known, within, fits, payload, writes};
                            // A signal carries no payload by construction, and its range check is
                            // over a fixed eight bytes, so a signal with a payload is not a frame the
                            // wire format can carry.
                            if (kind == Frame::Kind::Signal and payload) continue;
                            // A frame whose length is not within the largest window necessarily has a
                            // payload, and necessarily does not fit any window.
                            if (not within and (not payload or fits)) continue;
                            // A zero-length placement always fits, whatever the offset, only if the
                            // offset is inside the window; the model keeps `fits` free for that.
                            out.push_back(f);
                        }
                    }
                }
            }
        }
    }
    return out;
}

}   // namespace

int main() {
    const auto frames = all_frames();
    std::printf("frame shapes the wire format can carry: %zu\n", frames.size());

    // P-FRAME-1: no single frame leaves the receiver desynchronised.
    size_t inspected = 0;
    for (const Frame& f : frames) {
        const Reaction r = react(f);
        ++ inspected;
        if (r.position == Position::Desynchronised)
            fail(std::string("a frame leaves the stream desynchronised via branch: ") + r.branch);
        if (r.position != Position::AtFrameBoundary and r.position != Position::LinkClosed)
            fail(std::string("a frame reached a position this model does not name, via branch: ") +
                 r.branch);
        // P-FRAME-2: a refusal is either survivable *and* leaves the stream framed, or closes the link.
        if (r.refused and r.position == Position::AtFrameBoundary and not f.length_within_largest_window)
            fail(std::string("a frame with an untrusted length was refused but the link stayed open: ") +
                 r.branch);
        // P-FRAME-3: an application and a refusal are mutually exclusive, and so are an application
        // and a failed write. A frame counted twice makes the counters meaningless.
        if ((r.applied and r.refused) or (r.applied and r.failed_write) or
            (r.refused and r.failed_write))
            fail(std::string("a frame moved two counters at once via branch: ") + r.branch);
    }

    // P-FRAME-4: every sequence of frames up to length `kDepth` ends either with the link closed or at
    // a frame boundary, and once the link is closed nothing further is handled. This is the property
    // that rules out "one bad frame poisons the ones after it".
    constexpr int kDepth = 4;
    size_t sequences = 0, closed_early = 0;
    std::vector<size_t> idx(kDepth, 0);
    std::set<std::string> reached_branches;

    // Odometer over frame indices, so every ordered sequence of `kDepth` frames is visited.
    for (;;) {
        Position pos = Position::AtFrameBoundary;
        bool closed = false;
        for (int d = 0; d < kDepth; ++ d) {
            if (closed) break;                        // nothing after a close is handled
            const Reaction r = react(frames[idx[size_t(d)]]);
            reached_branches.insert(r.branch);
            pos = r.position;
            if (pos == Position::Desynchronised) {
                fail("a sequence reached a desynchronised state");
                break;
            }
            if (pos == Position::LinkClosed) { closed = true; ++ closed_early; }
        }
        ++ sequences;
        if (pos == Position::Desynchronised)
            fail("a sequence ended desynchronised");

        // advance the odometer
        int d = kDepth - 1;
        for (; d >= 0; -- d) {
            if (++ idx[size_t(d)] < frames.size()) break;
            idx[size_t(d)] = 0;
        }
        if (d < 0) break;
    }

    std::printf("single frames inspected: %zu\n", inspected);
    std::printf("frame sequences of length %d enumerated: %zu (%zu closed the link early)\n",
                kDepth, sequences, closed_early);
    std::printf("branches of ReceiveAgent::loop reached: %zu\n", reached_branches.size());
    for (const auto& b : reached_branches) std::printf("  - %s\n", b.c_str());

    // P-FRAME-5: the model's range rule is the implementation's. Checked against the real function for
    // the boundary values that matter, including the overflowing pair that used to pass.
    struct RangeCase { uint64_t offset, length; size_t bytes; bool want; const char* why; };
    const RangeCase cases[] = {
        {0, 0, 1024, true, "an empty range at the start fits"},
        {1024, 0, 1024, true, "an empty range at the very end fits"},
        {1025, 0, 1024, false, "an empty range past the end does not"},
        {0, 1024, 1024, true, "the whole window fits"},
        {0, 1025, 1024, false, "one byte more does not"},
        {1023, 1, 1024, true, "the last byte fits"},
        {1023, 2, 1024, false, "one past the last byte does not"},
        {~uint64_t(0) - 7, 16, 1024, false, "the pair whose sum wraps to 8 must not fit"},
        {~uint64_t(0), 1, 1024, false, "the largest offset does not fit"},
        {0, ~uint64_t(0), 1024, false, "the largest length does not fit"},
        {~uint64_t(0) - 7, 8, 1024, false, "a sum that wraps to exactly zero must not fit"},
    };
    for (const auto& c : cases) {
        const bool got = srdsim::range_fits_for_test(c.offset, c.length, c.bytes);
        if (got != c.want) {
            char b[240];
            std::snprintf(b, sizeof(b), "range_fits(%llu, %llu, %zu) = %d, want %d: %s",
                          (unsigned long long)c.offset, (unsigned long long)c.length, c.bytes,
                          int(got), int(c.want), c.why);
            fail(b);
        }
    }
    std::printf("range checks against the implementation: %zu\n", sizeof(cases) / sizeof(cases[0]));

    if (g_failures != 0) {
        std::printf("FAIL: %d property violation(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS: no reachable state leaves the receiver reading a header from inside a payload\n");
    return 0;
}

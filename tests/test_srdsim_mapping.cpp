// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// The two pieces of arithmetic that vLLM found and the suite did not.
//
// Both defects were in code that only compiles with `gdrapi.h` and only runs where `/dev/gdrdrv` exists,
// so the suite could not reach them; and both were invisible to the reproducer for a structural reason
// rather than by luck. The reproducer registers only page-aligned windows, so the lead offset was always
// zero. It runs three threads, so the last staging slot was never claimed. Neither fact is a property of
// the code -- they are properties of one harness -- and a test that depends on them is not a test.
//
// So these cases are enumerated: every slot rather than the ones a small run reaches, and lead offsets
// across a page rather than the one value an aligned allocation gives.

#include "srdsim_mapping.h"

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <initializer_list>

namespace {

int g_checks = 0;
int g_failures = 0;

void expect(bool ok, const char* what) {
    ++ g_checks;
    if (ok) return;
    ++ g_failures;
    std::printf("[FAIL] %s\n", what);
}

char g_ctx[400];
const char* ctx(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(g_ctx, sizeof(g_ctx), fmt, ap);
    va_end(ap);
    return g_ctx;
}

// M1. A window that begins on a page needs exactly its own length mapped, and one that does not needs
// more. The second case is the one that was wrong: the mapping was created with the window's length while
// accesses were offset by the lead, so the far end of every such window wrote past the mapping.
void test_the_mapping_covers_the_lead() {
    const std::size_t page = srdsim::kGpuPageBytes;

    {
        const auto s = srdsim::mapping_shape(0x1000000, 0x1000000, 2048, page);
        expect(s.valid, "M1 an aligned window is mappable");
        expect(s.lead == 0, "M1 an aligned window has no lead");
        expect(s.map_bytes == page, "M1 a 2048 byte aligned window still maps a whole page");
    }

    // Every offset within a page, for a window small enough to fit and one large enough not to. The far
    // end of the window has to be inside the mapping in both cases.
    for (std::size_t lead = 0; lead < page; lead += 761) {
        for (std::size_t bytes : {std::size_t(8), std::size_t(1848), std::size_t(73856),
                                  std::size_t(2u << 20)}) {
            const std::uintptr_t va = 0x40000000;
            const auto s = srdsim::mapping_shape(va + lead, va, bytes, page);
            expect(s.valid, ctx("M1 offset %zu is mappable", lead));
            expect(s.lead == lead, ctx("M1 the lead is %zu", lead));
            expect(s.map_bytes >= lead + bytes,
                   ctx("M1 the mapping covers lead %zu plus %zu byte(s) (mapped %zu)",
                       lead, bytes, s.map_bytes));
            expect(s.map_bytes % page == 0,
                   ctx("M1 the mapping is a whole number of pages (lead %zu, %zu byte(s))", lead, bytes));
        }
    }

    // A pinned range starting above the window cannot happen, and the arithmetic must refuse it rather
    // than underflow into an enormous lead.
    const auto bad = srdsim::mapping_shape(0x40000000, 0x40001000, 4096, page);
    expect(not bad.valid, "M1 a pinned range above the window is refused");
    expect(bad.lead == 0 and bad.map_bytes == 0, "M1 a refused shape carries no numbers");
}

// M2. The offset an access lands at, and the bound it has to satisfy. The window is the authority, not the
// mapping: page rounding makes the mapping longer, and writing into that slack would put bytes past the
// memory NCCL registered.
void test_an_access_is_bounded_by_the_window_not_the_mapping() {
    const std::size_t page = srdsim::kGpuPageBytes;
    const std::uintptr_t va = 0x80000000;
    const std::size_t lead = 4096;
    const std::uintptr_t base = va + lead;
    const std::size_t bytes = 1848;

    std::size_t off = ~std::size_t(0);
    expect(srdsim::mapping_offset(base, bytes, va, base, 8, &off), "M2 the first eight bytes are inside");
    expect(off == lead, ctx("M2 the first access lands at the lead (%zu)", off));

    expect(srdsim::mapping_offset(base, bytes, va, base + bytes - 8, 8, &off),
           "M2 the last eight bytes are inside");
    expect(off == lead + bytes - 8, ctx("M2 the last access lands at %zu", off));

    // One byte past the window is outside, even though the mapping is a whole page and physically has
    // room. This is the check that stops a write from leaving the registered region.
    expect(not srdsim::mapping_offset(base, bytes, va, base + bytes, 1, nullptr),
           "M2 one byte past the window is refused although the mapping is longer");
    expect(not srdsim::mapping_offset(base, bytes, va, base + bytes - 4, 8, nullptr),
           "M2 an access straddling the end is refused");
    expect(not srdsim::mapping_offset(base, bytes, va, base - 1, 1, nullptr),
           "M2 an access below the window is refused");

    // Values chosen so that `offset + length` wraps. Written as a subtraction there is nothing to wrap,
    // and the case has to stay refused rather than pass by arithmetic accident.
    expect(not srdsim::mapping_offset(base, bytes, va, base + 8, ~std::size_t(0) - 4, nullptr),
           "M2 a length that would wrap the addition is refused");
    (void)page;
}

// M3. No staging slot shares a byte with the signal slot. Asserted for every slot, because the defect was
// in the last one and a three-thread run never reaches it.
void test_no_staging_slot_touches_the_signal() {
    for (std::size_t slots : {std::size_t(1), std::size_t(2), std::size_t(63), std::size_t(64)}) {
        for (std::size_t slot_bytes : {std::size_t(8), std::size_t(4096), std::size_t(8192)}) {
            const srdsim::PoolLayout p{slots, slot_bytes};
            expect(p.total_bytes() == slot_bytes * (slots + 1),
                   ctx("M3 the pool holds one slot more than it hands out (%zu slots of %zu)",
                       slots, slot_bytes));
            for (std::size_t i = 0; i < slots; ++ i)
                expect(p.signal_is_disjoint_from(i, sizeof(std::uint64_t)),
                       ctx("M3 slot %zu of %zu does not touch the signal slot (%zu byte slots)",
                           i, slots, slot_bytes));
            expect(p.signal_offset() >= p.staging_offset(slots - 1) + slot_bytes,
                   ctx("M3 the signal starts at or after the end of the last staging slot "
                       "(%zu slots of %zu)", slots, slot_bytes));
            // And the signal's own eight bytes are inside the pool rather than one past it.
            expect(p.signal_offset() + sizeof(std::uint64_t) <= p.total_bytes(),
                   ctx("M3 the signal fits inside the pool (%zu slots of %zu)", slots, slot_bytes));
        }
    }

    // Every slot of every group, including the send group, stays in its own group, and no slot's bytes
    // reach the signal entries; every signal entry lies inside the pool.
    for (const srdsim::StagingShape shape : {srdsim::StagingShape{32, 2}, srdsim::StagingShape{32, 128}}) {
        const srdsim::PoolLayout p{shape.slots(), 8192};
        for (std::size_t g = 0; g < shape.groups; ++ g)
            for (std::size_t i = 0; i < 2 * shape.ring; ++ i) {
                const std::size_t slot = shape.slot(g, i);
                expect(slot / shape.ring == g and slot < p.slots,
                       ctx("M3 ring %zu: slot %zu of group %zu is %zu, inside its group", shape.ring, i, g, slot));
                expect(p.signal_is_disjoint_from(slot, p.slot_bytes),
                       ctx("M3 ring %zu: no byte of slot %zu reaches the signal entries", shape.ring, slot));
            }
        expect(shape.send_group() == shape.groups - 1 and shape.receive_groups() == shape.groups - 1,
               ctx("M3 ring %zu: the send group is the last and no receiving thread can claim it", shape.ring));
        for (std::size_t k = 0; k < 2 * p.signal_entries(); ++ k)
            expect(p.signal_entry_offset(k) >= p.signal_offset() and
                   p.signal_entry_offset(k) + sizeof(std::uint64_t) <= p.total_bytes(),
                   ctx("M3 ring %zu: signal entry %zu lies inside the pool", shape.ring, k));
    }

    // The shape the plugin actually uses, stated once so a change to it is visible here.
    const srdsim::PoolLayout used{64, 8192};
    expect(used.total_bytes() == 532480, ctx("M3 the plugin's pool is 532480 bytes (%zu)",
                                             used.total_bytes()));
    expect(used.signal_offset() == 524288, ctx("M3 the signal sits at 524288 (%zu)",
                                               used.signal_offset()));
}

// M4. Rounding up to a page is exact at the boundaries, which is where an off-by-one would hide.
void test_rounding_is_exact_at_the_boundaries() {
    const std::size_t page = srdsim::kGpuPageBytes;
    expect(srdsim::round_up_to_page(0, page) == 0, "M4 nothing rounds to nothing");
    expect(srdsim::round_up_to_page(1, page) == page, "M4 one byte rounds to one page");
    expect(srdsim::round_up_to_page(page, page) == page, "M4 a whole page stays one page");
    expect(srdsim::round_up_to_page(page + 1, page) == 2 * page, "M4 one byte over rounds to two pages");
    expect(srdsim::round_up_to_page(2 * page, page) == 2 * page, "M4 two whole pages stay two");
    expect(srdsim::round_up_to_page(1822425088, page) == 1822425088,
           "M4 the 1.82 GB window is already a whole number of pages");
    expect(srdsim::round_up_to_page(1848, page) == page, "M4 the signal window rounds to one page");
    expect(srdsim::round_up_to_page(73856, page) == 2 * page, "M4 73856 bytes needs two pages");
}

}  // namespace

int main() {
    test_the_mapping_covers_the_lead();
    test_an_access_is_bounded_by_the_window_not_the_mapping();
    test_no_staging_slot_touches_the_signal();
    test_rounding_is_exact_at_the_boundaries();
    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks,
                g_failures);
    return g_failures == 0 ? 0 : 1;
}

// SPDX-License-Identifier: Apache-2.0

// The address arithmetic behind a BAR mapping, and the layout of the pinned staging pool.
//
// WHY THIS IS A SEPARATE HEADER
// Both are easy to get wrong in ways a small run does not reach:
//
//   * a signal's eight bytes carved out of the end of the pool lie inside the last staging slot, so a
//     thread holding that slot and the signal path write the same bytes. A run with only a few threads
//     never reaches the last slot;
//   * a mapping created with the window's length, while accesses are offset by the distance from the
//     pinned page to the window, runs off its end by exactly that distance. A window that begins on a
//     page has a distance of zero.
//
// Neither needs a GPU to be wrong or to be checked, but the code that uses them only compiles with
// `gdrapi.h` and only runs where `/dev/gdrdrv` exists. Pulled out here it is ordinary integer arithmetic
// over `uintptr_t`, so the suite can enumerate the cases that matter -- offset windows, page boundaries, the
// last slot, and values
// chosen to overflow if the arithmetic is written the obvious way.
//
// The rule these follow, and the reason they are written the way they are: no expression here adds two
// values that came from outside. `base + bytes` overflows for a large enough pair and then compares as
// small, which is not a bounds check. Subtraction from a known-larger value cannot.

#ifndef SRDSIM_MAPPING_H_
#define SRDSIM_MAPPING_H_

#include <cstddef>
#include <cstdint>

namespace srdsim {

// The size of the GPU page GDRCopy pins in. Named rather than taken from `gdrapi.h` so this header has no
// dependency on it, and asserted against the real value where the two meet.
inline constexpr std::size_t kGpuPageBytes = 65536;

// ---------------------------------------------------------------------------
// One mapped window
// ---------------------------------------------------------------------------

// Where a window sits inside the range that was actually pinned.
//
// `pinned_va` is what `gdr_get_info` reports: the start of the GPU page containing the window, which is at
// or below the window's own base. `lead` is the difference, and every access through the mapping carries
// it. A mapping that does not include the lead is shorter than the accesses made through it.
struct MappingShape {
    bool     valid = false;      // false when the pinned range starts above the window, which cannot be
    std::size_t lead = 0;        // bytes from the pinned page to the window base
    std::size_t map_bytes = 0;   // what has to be mapped to cover lead + window
};

// Rounds up to a whole page without adding two outside values.
inline std::size_t round_up_to_page(std::size_t bytes, std::size_t page = kGpuPageBytes) {
    if (page == 0) return bytes;
    const std::size_t remainder = bytes % page;
    if (remainder == 0) return bytes;
    // `bytes - remainder` cannot overflow, and adding one page to it only can if `bytes` was already
    // within a page of the maximum, which no allocation is.
    return (bytes - remainder) + page;
}

inline MappingShape mapping_shape(std::uintptr_t window_base, std::uintptr_t pinned_va,
                                  std::size_t window_bytes, std::size_t page = kGpuPageBytes) {
    MappingShape s{};
    if (pinned_va > window_base) return s;      // would underflow; the caller must not map this
    s.lead = static_cast<std::size_t>(window_base - pinned_va);
    s.map_bytes = round_up_to_page(s.lead + window_bytes, page);
    s.valid = true;
    return s;
}

// The offset into the mapping at which `access` begins, or `false` when the access is not wholly inside
// the window.
//
// The window is the authority on what may be touched, not the mapping: the mapping can be longer because
// of page rounding, and writing into that slack would put bytes past the end of the memory NCCL
// registered.
inline bool mapping_offset(std::uintptr_t window_base, std::size_t window_bytes,
                          std::uintptr_t pinned_va, std::uintptr_t access, std::size_t access_bytes,
                          std::size_t* offset_out) {
    if (access < window_base) return false;
    const std::size_t within = static_cast<std::size_t>(access - window_base);
    if (within > window_bytes) return false;
    if (access_bytes > window_bytes - within) return false;   // no addition, so nothing can wrap
    if (pinned_va > window_base) return false;
    const std::size_t lead = static_cast<std::size_t>(window_base - pinned_va);
    if (offset_out != nullptr) *offset_out = lead + within;
    return true;
}

// ---------------------------------------------------------------------------
// The pinned staging pool
// ---------------------------------------------------------------------------

// The pool holds one slot per thread that stages bytes, plus one more for the signal totals. The signal
// slot is past every staging slot rather than carved out of the last one, which is the defect this
// layout exists to make impossible to reintroduce silently.
// How the staging pool is carved up. `groups` groups of `ring` slots: every group but the last belongs to one
// receiving thread, which copies from its slots in turn; the last group is the send side's. After the slots
// comes one more slot's worth of 8-byte signal entries, used in turn as well.
//
// In a run where every delivery is a CUDA copy from the plugin's own context, a copy is waited for only when
// its slot comes round again, so `ring` sets how many copies can be outstanding per thread. With GDRCopy the
// copies are rare fallbacks and are waited for at once, and a ring of 2 is enough.
struct StagingShape {
    std::size_t groups = 0;
    std::size_t ring = 0;
    std::size_t slot(std::size_t group, std::size_t i) const { return group * ring + i % ring; }
    std::size_t send_group() const { return groups - 1; }
    std::size_t receive_groups() const { return groups - 1; }
    std::size_t slots() const { return groups * ring; }
};

struct PoolLayout {
    std::size_t slots = 0;
    std::size_t slot_bytes = 0;

    std::size_t total_bytes() const { return slot_bytes * (slots + 1); }
    std::size_t staging_offset(std::size_t slot) const { return slot * slot_bytes; }
    std::size_t signal_offset() const { return slots * slot_bytes; }
    std::size_t signal_entries() const { return slot_bytes / sizeof(std::uint64_t); }
    std::size_t signal_entry_offset(std::size_t k) const {
        return signal_offset() + (k % signal_entries()) * sizeof(std::uint64_t);
    }

    // True when a staging slot and the signal slot can never touch the same byte. Asserted by the suite
    // for every slot rather than reasoned about, because the reasoning is what was wrong.
    bool signal_is_disjoint_from(std::size_t slot, std::size_t signal_bytes) const {
        if (slot >= slots) return false;
        const std::size_t staging_end = staging_offset(slot) + slot_bytes;
        const std::size_t signal_start = signal_offset();
        if (staging_end > signal_start) return false;
        return signal_bytes <= total_bytes() - signal_start;
    }
};

}  // namespace srdsim

#endif  // SRDSIM_MAPPING_H_

// Without-replacement position sampler (SRSWOR core).
//
// The sampler draws a requested number of NEW stratum-local positions
// uniformly at random, excluding positions already recorded in a tracker, so
// that cumulative rounds never re-draw a row. The algorithm is enumerate +
// partial Fisher-Yates: list the eligible positions, then shuffle just enough
// of the front of that list to expose the requested count. Enumeration is
// linear in the eligible population; the shuffle is linear in the draw size.
//
// Returned positions are in shuffle order, not sorted; ordering for read
// locality happens later where the rows are actually read.
//
// PROVISIONAL INTERFACE. These two entry points always
// materialize the eligible set into a vector and enumerate it, which costs
// O(eligible population) per draw even for a tiny sample of a large
// fully-contained partition. That is correct but not how the eligible
// universe is ultimately represented:
//   * a fully-contained partition needs no candidate list at all -- its
//     eligible universe is implicitly [0, size) and a draw can reject random
//     positions against the tracker in O(draw) without an array;
//   * a partially-contained partition's qualifying positions are produced by
//     the geometric decomposition pass as a position bitset (one bit per
//     local position), which supports both O(1) membership (for rejection)
//     and enumeration (for Fisher-Yates) -- strictly more capable than the
//     dense `candidates` span this header currently takes.
// The planned replacement is a single `draw(EligibleUniverse, tracker, count,
// rng)` where `EligibleUniverse = {size, const PositionBitset* qualifying}`
// (qualifying == nullptr means fully contained), choosing rejection vs
// enumerate+Fisher-Yates by the draw-size-to-eligible ratio. It lands together
// with the qualifying-bitset producer in the decomposition pass, which also
// provides the query path that exercises the fast path. Until then the
// span-based `draw_from_bitset` and the enumerate-always `draw_from_range`
// here are the simple, correct version.

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "a3i/aqp/summary.hpp"
#include "a3i/core/types.hpp"
#include "a3i/util/rng.hpp"

namespace a3i {

struct Sampler {
    /// Draw `count` distinct positions from [0, size) excluding those already
    /// in `tracker`. If fewer than `count` remain eligible, returns them all.
    static std::vector<IndexPos> draw_from_range(std::uint64_t size,
                                                 const SampleTracker& tracker,
                                                 std::uint64_t count, Rng& rng);

    /// Draw `count` distinct positions from `candidates` excluding those
    /// already in `tracker`. If fewer than `count` remain eligible, returns
    /// them all.
    static std::vector<IndexPos> draw_from_bitset(
        std::span<const IndexPos> candidates, const SampleTracker& tracker,
        std::uint64_t count, Rng& rng);
};

}  // namespace a3i

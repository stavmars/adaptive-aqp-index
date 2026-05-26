// Without-replacement position sampler (SRSWOR core).
//
// The sampler draws a requested number of NEW positions uniformly at random
// from an eligible universe, excluding positions already recorded in a
// tracker, so that cumulative rounds never re-draw a row. The eligible
// universe is a contiguous range [0, size) optionally narrowed to a
// qualifying subset (a position bitset); when no subset is given the whole
// range qualifies.
//
// Two strategies give identical SRSWOR semantics; the sampler picks whichever
// is cheaper for the draw at hand:
//   * rejection -- propose uniform positions in [0, size) and reject those
//     outside the qualifying set, already in the tracker, or already chosen
//     this draw. O(draw) with no allocation proportional to the universe;
//     used when the draw is a small fraction of a sparsely-sampled, dense
//     eligible set.
//   * enumerate + partial Fisher-Yates -- list the eligible positions, then
//     shuffle just enough of the front to expose the requested count. Linear
//     in the eligible population; used when the draw is large relative to the
//     eligible set or the set is sparse (where rejection would thrash).
// The two strategies are not seed-compatible with each other, but the choice
// is a deterministic function of the draw and the eligible size, so a given
// draw is reproducible for a given build.
//
// Returned positions are in draw order, not sorted; ordering for read
// locality happens later where the rows are actually read.

#pragma once

#include <cstdint>
#include <vector>

#include "a3i/aqp/position_bitset.hpp"
#include "a3i/aqp/summary.hpp"
#include "a3i/core/types.hpp"
#include "a3i/util/rng.hpp"

namespace a3i {

/// The set a draw samples from: positions [0, size), narrowed to `qualifying`
/// when non-null. A null `qualifying` means every position qualifies (a
/// fully-contained partition), so no candidate list is needed at all.
struct EligibleUniverse {
    std::uint64_t         size       = 0;
    const PositionBitset* qualifying = nullptr;
};

struct Sampler {
    /// Draw `count` distinct positions from `universe` excluding those already
    /// in `tracker`. If fewer than `count` remain eligible, returns them all.
    /// Positions share the coordinate space of `tracker` and the qualifying
    /// bitset (offsets into the partition's [0, size) positions).
    static std::vector<IndexPos> draw(EligibleUniverse universe,
                                      const SampleTracker& tracker,
                                      std::uint64_t count, Rng& rng);
};

}  // namespace a3i

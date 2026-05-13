// Lightweight views the access path exposes to the query layer.
//
// A PartitionView is a read-only snapshot of one partition: its dense id,
// its spatial bounds, the contiguous [begin, end) slice of the index table
// it owns, and whether it is currently active (a leaf of the live
// structure) or retired (a former parent kept for ancestry). A
// QueryPartitionSet is the result of locating a query rectangle: the
// active partitions fully inside the rectangle and those only partially
// overlapping it. Disjoint partitions are simply absent from both lists. A
// RefineResult bundles the outcome of a refinement: the parents that
// retired during cracking, plus the post-refinement classification of the
// active frontier against the query, so the caller does not have to walk
// the structure a second time to re-derive it.

#pragma once

#include <vector>

#include "a3i/core/geometry.hpp"
#include "a3i/core/types.hpp"

namespace a3i {

struct PartitionView {
    PartitionId id    = 0;
    HyperRect   bounds;
    IndexPos    begin = 0;
    IndexPos    end   = 0;
    bool        active = true;
};

struct QueryPartitionSet {
    std::vector<PartitionId> fully_contained;
    std::vector<PartitionId> partial;
};

struct RefineResult {
    // Parents that retired during this refinement (now inactive ancestors
    // whose summaries the caller retains).
    std::vector<PartitionId> retired;
    // The active frontier classified against the query after refinement.
    // Equivalent to calling locate(q) on the refined structure, but produced
    // as a by-product of the descent refine already performs.
    QueryPartitionSet frontier;
};

}  // namespace a3i

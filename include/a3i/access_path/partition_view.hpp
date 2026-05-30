// Lightweight views the access path exposes to the query layer.
//
// A PartitionView is a read-only snapshot of one partition: its dense id,
// its spatial bounds, the contiguous [begin, end) slice of the index table
// it owns, and whether it is currently active (a leaf of the live
// structure) or retired (a former parent kept for ancestry).
//
// Containment classifies one partition's bounds against a query rectangle and
// is pure geometry: Disjoint when the bounds and the query do not overlap,
// Contained when the query encloses the whole bounds, Partial otherwise. The
// engine's top-down descent uses it at each node to decide whether to drop,
// stop, or recurse.

#pragma once

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

enum class Containment { Disjoint, Contained, Partial };

}  // namespace a3i

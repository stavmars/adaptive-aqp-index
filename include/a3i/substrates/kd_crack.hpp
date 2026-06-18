// Query-driven cracking of a KD partitioning shared by the substrates that
// adapt under queries.

#pragma once

#include <cstdint>
#include <vector>

#include "a3i/core/geometry.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/kd_tree.hpp"

namespace a3i {

/// Crack the leaf `id` in `tree` toward `q`: split at each axis' lower bound
/// (descending into the >= child) then at each upper bound (descending into the
/// < child), stopping as soon as the surviving boundary child is no larger than
/// `max_population`. Appends every retired parent id to `retired` and returns
/// the id of the surviving boundary child (== `id` when no split made progress).
/// Every discarded child lies wholly outside `q`. Never reads measures.
PartitionId crack_to_query(KdTree& tree, IndexTable& table, PartitionId id,
                           const HyperRect& q, std::uint32_t max_population,
                           std::vector<PartitionId>& retired);

}  // namespace a3i

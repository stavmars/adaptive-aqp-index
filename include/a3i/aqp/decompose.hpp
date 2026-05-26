// Single-pass query decomposition.
//
// Given the post-refinement frontier (the active partitions classified as
// fully contained or partially overlapping a query), split the query's
// qualifying objects into the disjoint contributors of a query answer and
// compute, for free, the deterministic exact totals and the COUNT(*) of all
// qualifying objects.
//
// The classification of geometry is done once here, not per measure: a
// partition is fully contained or partial regardless of which measure is
// asked. A fully-contained partition is an exact contributor only if every
// measure is already exact for it (all-or-nothing); otherwise it is a
// reusable stratum sampled for all measures together over one shared tracker.
// A partial partition is scanned once to build its qualifying-position bitset,
// shared across the partition's measures.

#pragma once

#include <cstddef>

#include "a3i/access_path/adaptive_access_path.hpp"
#include "a3i/access_path/partition_view.hpp"
#include "a3i/aqp/partition_state_store.hpp"
#include "a3i/aqp/query_decomposition.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/storage/index_table.hpp"

namespace a3i {

/// Decompose query `q` over the already-classified `frontier`. `state_store`
/// is consulted (and, for sampled partitions, has empty summaries created in
/// it) so reusable strata carry their partition's persistent tracker. No
/// measure values are read: exact totals come from stored summaries and the
/// qualifying count from a dimension-only scan of the partial partitions.
DecompositionResult decompose(const HyperRect& q,
                              const AdaptiveAccessPath& access_path,
                              const QueryPartitionSet& frontier,
                              PartitionStateStore& state_store,
                              const IndexTable& table,
                              std::size_t measure_count);

}  // namespace a3i

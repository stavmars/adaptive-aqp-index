// One-shot exact summary fill for a fully-built, fixed (non-adaptive)
// partitioning.
//
// Given an access path whose whole structure already exists -- every active
// partition owns a fixed contiguous range that never moves and never splits --
// this precomputes an exact aggregate summary for every partition, both the
// active partitions and every parent partition above them. A later query fully
// inside an active partition answers from its stored summary with no query-time
// measure reads; a query that fully contains a whole parent partition answers
// from that one parent summary instead of descending into its sub-partitions.
//
// The fill is I/O-shaped: each measure column is read exactly once in ascending
// storage (row-id) order. Because the build permutes rows, partition order is
// not row-id order, so a transient owner[row_id] -> partition map routes each
// sequentially-read value into its owning partition's accumulator. The only
// random access is the small write into the per-partition accumulator set; the
// large measure data is streamed sequentially -- each value read once, in order.
//
// Active-partition summaries are exact and additive, so any parent summary is
// the merge of its descendants and needs no further measure reads: this routine
// fills the active partitions by the single column sweep, then folds those exact
// moments upward into every parent partition for free.

#pragma once

#include <cstddef>
#include <vector>

#include "a3i/access_path/adaptive_access_path.hpp"
#include "a3i/aqp/partition_state_store.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/binary_column_store.hpp"
#include "a3i/storage/index_table.hpp"

namespace a3i {

class OutlierScorer;

/// Fill an exact, complete summary for every active partition of `access_path`
/// and for every parent partition above them into `state`. `table` provides the
/// row id at each position; `store` provides the measure values. Each of the
/// `measure_count` measure columns is read once in ascending row-id order.
/// Precondition: `access_path.ensure_built()` has run.
///
/// `owner`, when non-null, supplies the RowId -> active-partition map directly
/// (size must equal table.size()), skipping its derivation from the partition
/// ranges; when null the map is built by walking those ranges.
///
/// `scorer`, when non-null, is fed every row's measure values during the single
/// column sweep, so the outlier index is built from the same reads with no
/// extra pass.
void materialize_all_summaries(const AdaptiveAccessPath& access_path,
                               const IndexTable& table,
                               const BinaryColumnStore& store,
                               PartitionStateStore& state,
                               std::size_t measure_count,
                               const std::vector<PartitionId>* owner = nullptr,
                               OutlierScorer* scorer = nullptr);

}  // namespace a3i

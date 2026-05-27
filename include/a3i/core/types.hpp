// Identifier types used across the index, storage, and aggregation layers.
//
// Widths are chosen by what each id counts, not lumped together:
//   - RowId / IndexPos are 32-bit (datasets target up to ~10^9 rows;
//     widen together to 64-bit if a future dataset exceeds ~4.29B rows).
//   - PartitionId is 32-bit (dense, consecutive, never reused).
//   - MeasureId / DimensionId are 16-bit (they count columns, not rows).

#pragma once

#include <cstdint>

namespace a3i {

/// Stable base-table row ordinal; the key used to gather measure values.
using RowId = std::uint32_t;

/// Position within the mutable in-memory index table.
using IndexPos = std::uint32_t;

/// Dense, consecutive partition id assigned by the access path.
using PartitionId = std::uint32_t;

/// Index into the schema's measure columns.
using MeasureId = std::uint16_t;

/// Index into the schema's dimension columns.
using DimensionId = std::uint16_t;

/// Per-round index identifying a stratum within one sampling round; used to
/// route a gathered measure value back to its accumulators. Local to a
/// round, no persistent meaning. 32-bit because one query can touch as many
/// strata as it has residual partitions, which fine-grained cracking can push
/// past a 16-bit ceiling.
using StratumTag = std::uint32_t;

}  // namespace a3i

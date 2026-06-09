// The disjoint split of a query's qualifying objects into contributors.
//
// A query answer is built from contributors that partition the qualifying
// objects with no overlap (no row counted twice). Each kind differs by how
// its value is obtained and by how long its state lives:
//
//   * ExactContributor -- a fully-contained partition whose summary is
//     already exact for the measure. Contributes a deterministic value with
//     zero variance and needs no reads.
//   * ReusableStratum -- a fully-contained partition not yet exact. Sampled
//     over its whole [begin, end); its summary and sample tracker are
//     persistent, so later queries reuse the progress.
//   * QueryLocalStratum -- a partially-overlapping partition. Only the
//     qualifying positions take part; the sample is query-local and its
//     tracker is discarded when the query ends.
//   * QueryLocalExactContributor -- a query-local stratum that was fully read
//     during exactification, so it now contributes exactly.
//
// Geometry (which positions qualify) is measure-independent, so a partition's
// qualifying set is shared across its measures rather than copied per measure.
// Sampling progress (the tracker) is likewise one per partition.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "a3i/aqp/position_bitset.hpp"
#include "a3i/aqp/summary.hpp"
#include "a3i/core/types.hpp"

namespace a3i {

struct ExactContributor {
    PartitionId           pid = 0;
    MeasureId             mid = 0;
    const MeasureSummary* summary = nullptr;
    bool                  retained_ancestor = false;
};

struct ReusableStratum {
    PartitionId   pid = 0;
    MeasureId     mid = 0;
    IndexPos      begin = 0;
    IndexPos      end   = 0;
    std::uint64_t population_size = 0;
    std::shared_ptr<SampleTracker> tracker;  // persistent, shared by the partition's measures
};

struct QueryLocalStratum {
    PartitionId pid = 0;
    MeasureId   mid = 0;
    IndexPos    begin = 0;  // partition start in the index table
    // Partition-local qualifying positions; shared across the partition's
    // measures (geometry is measure-independent).
    std::shared_ptr<const PositionBitset> qualifying;
    std::uint64_t population_size = 0;  // == qualifying->count()
};

struct QueryLocalStratumState {
    std::uint64_t sampled_rows = 0;
    MomentStats   non_nan;
    std::shared_ptr<SampleTracker> tracker;  // fresh per query, discarded at end
};

struct QueryLocalExactContributor {
    PartitionId pid = 0;
    MeasureId   mid = 0;
    IndexPos    begin = 0;
    std::shared_ptr<const PositionBitset> qualifying;
};

struct QueryDecomposition {
    std::vector<ExactContributor>           exact_contributors;
    std::vector<ReusableStratum>            reusable_strata;
    std::vector<QueryLocalStratum>          query_local_strata;
    std::vector<QueryLocalExactContributor> query_local_exact_contributors;
};

/// Per-measure deterministic totals from the exact contributors: the summed
/// non-missing values and the count of non-missing values that fed them.
struct ExactBucket {
    std::vector<double>        sum_by_measure;
    std::vector<std::uint64_t> count_by_measure;
};

struct DecompositionResult {
    QueryDecomposition decomposition;
    ExactBucket        exact_bucket;
    std::uint64_t      total_count = 0;  // COUNT(*): all qualifying objects, incl. missing
    std::uint64_t      nodes_visited = 0;       // descent nodes classified (incl. dropped)
    std::uint64_t      partitions_refined = 0;   // boundary leaves cracked toward the query (>=1 cut each)
    std::uint64_t      frontier_partitions = 0;  // leaf-level partitions emitted as contributors/strata
    // Per-partition breakdown of the frontier; these four sum to
    // `frontier_partitions`. Counted once per partition, never per measure.
    std::uint64_t      exact_contributor_partitions = 0;  // complete (exact) prior summary
    std::uint64_t      reusable_sampled_partitions  = 0;  // partial (sampled) prior summary
    std::uint64_t      reusable_absent_partitions   = 0;  // no summary yet
    std::uint64_t      query_local_partitions       = 0;  // boundary partition
};

}  // namespace a3i

// Query request and result types.
//
// A query is a viewport rectangle plus the accuracy the caller wants for
// THIS query. Accuracy is a property of the request, not of the index: the
// same index can answer one query exactly and the next approximately. Which
// measures exist is fixed by the DatasetSchema the index was built with; all
// aggregates (SUM, COUNT(measure), AVG, COUNT(*)) are always computed for
// every query over every schema measure. There is no per-query measure or
// aggregate selection.
//
// Exact and approximate answers share ONE result type: an exact estimate is
// just an approximate one with a zero-width interval (`ci_low == ci_high ==
// estimate`, `relative_half_width == 0`, `exact == true`). A query that was
// requested approximate can end up exactified, and a single type represents
// that without conversion.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "a3i/core/geometry.hpp"
#include "a3i/core/types.hpp"

namespace a3i {

enum class AggregateOp { Sum, CountMeasure, Avg, CountStar };

/// What accuracy the caller wants for one query. `relative_error <= 0`
/// means the caller wants an exact answer; `> 0` is the relative
/// half-width target each aggregate's interval must meet at `confidence`.
struct AccuracyTarget {
    double relative_error = 0.0;
    double confidence     = 0.95;
};

/// One range-aggregate query: a predicate rectangle plus its accuracy
/// target. There is no query id in the logical model; the experiment
/// runner assigns an ordinal for logging only.
struct RangeQuery {
    HyperRect      predicate;
    AccuracyTarget target;
};

// Future work (per-query measure subsets): support either one sample bitset
// per (partition, measure) so measures sample independently, or one bitset
// per partition that "catches up" a newly requested measure by reading it
// for the rows already sampled before drawing new ones. Until then, an
// index's measure set is fixed by its DatasetSchema and every query computes
// every aggregate over every measure.

/// One aggregate's answer. For an exact answer the interval is degenerate:
/// `ci_low == ci_high == estimate`, `relative_half_width == 0`, `exact`.
struct AggregateEstimate {
    AggregateOp op;
    MeasureId   measure_id = 0;
    double      estimate            = 0.0;
    double      ci_low              = 0.0;
    double      ci_high             = 0.0;
    double      relative_half_width = 0.0;
    double      effective_df        = 0.0;
    bool        exact               = false;
};

/// Per-query bookkeeping: timings, work counters, and the outcome
/// taxonomy. Most fields are filled by the query engine; the exact-scan
/// oracle fills only what a full scan can report. Every field below is also
/// a column in the results CSV (`cell_runner.cpp` `kHeader`), except
/// `stop_reason`, which is an internal note that is not emitted.
struct QueryMetrics {
    /// 0-based position of this query within its workload.
    std::uint64_t query_ordinal = 0;
    /// Run name, e.g. "a3i" / "scan".
    std::string   method;
    /// Substrate id (e.g. "adaptive_kd"), "n/a" for the scan oracle.
    std::string   substrate;
    /// Outcome taxonomy: "exact" (exact-requested path), "converged"
    /// (sampling alone met the target), "exactified" (sampling stopped and
    /// the residual was read in full, then the target was met), or
    /// "exhausted_unconverged" (the round budget was spent unmet).
    std::string   status;
    /// Internal free-form note; NOT emitted to the results CSV.
    std::string   stop_reason;
    // Why the residual was read in full, if it was: "none", when no
    // exactification happened; "cheaper_to_exactify", when the next sampling
    // round would have read more than half the remaining residual; or
    // "gave_up", when planning stopped making progress or the round budget
    // was spent. Recorded after the fact for analysis; it does not steer the
    // control flow.
    std::string   exactify_cause = "none";

    /// Wall-clock of `execute()` for this query; filled by the runner.
    double        latency_ms = 0.0;

    // --- Read-work counters --------------------------------------------------
    // Sampling and exactification draw disjoint rows (without-replacement
    // sampling never re-reads a row), so they are counted separately and the
    // following identity holds for every method:
    //   measure_reads == (sampled_rows + exactified_rows) * measure_count.
    /// Total measure values gathered by row id (rows x measures): the I/O cost.
    std::uint64_t measure_reads    = 0;
    /// Distinct rows read by without-replacement sampling (tightening rounds).
    std::uint64_t sampled_rows     = 0;
    /// Distinct rows read by exhausting a stratum to exact (selective
    /// exactification; for the scan oracle, every qualifying row).
    std::uint64_t exactified_rows  = 0;

    // --- Decomposition frontier (per-partition counts) -----------------------
    // The descent stops at a frontier of partitions and emits each as one
    // contributor or stratum. These are counted ONCE PER PARTITION (not per
    // (partition, measure)); the four contributor categories below sum to
    // `frontier_partitions`.
    /// Partitions on the decomposition frontier (where the descent stops and
    /// emits a contributor/stratum). Excludes interior nodes passed through
    /// and disjoint nodes dropped.
    std::uint64_t frontier_partitions = 0;
    /// Boundary leaves the descent cracked toward this query (each counted
    /// once if it produced at least one cut). A measure of cracking work.
    std::uint64_t partitions_refined  = 0;
    /// Contained partitions answered from a complete (exact) prior summary;
    /// no reads. Grows over a session as reuse accumulates.
    std::uint64_t exact_contributors  = 0;
    /// Contained leaves carrying a partial (sampled) prior summary; sampling
    /// resumes cumulatively from where an earlier query left off.
    std::uint64_t reusable_sampled_strata = 0;
    /// Contained leaves with no summary yet (freshly cracked or never
    /// sampled); sampled from scratch.
    std::uint64_t reusable_absent_strata  = 0;
    /// Boundary partitions: only the in-rectangle subset qualifies and its
    /// samples are discarded when the query ends.
    std::uint64_t query_local_strata      = 0;

    /// Number of sampling/exactify rounds the engine loop ran.
    std::uint64_t adaptive_rounds    = 0;

    /// Did every requested aggregate meet its accuracy target?
    bool   target_satisfied = false;
    /// Worst relative half-width just before exactification kicked in; 0 when
    /// no exactification happened.
    double pre_exactification_error_bound = 0.0;
    /// Seed material for this run (the run id).
    std::uint64_t sampling_seed = 0;
};

struct QueryResult {
    std::vector<AggregateEstimate> aggregates;
    QueryMetrics                   metrics;
};

}  // namespace a3i

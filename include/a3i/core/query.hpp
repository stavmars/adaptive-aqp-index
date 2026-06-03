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
/// oracle fills only what a full scan can report.
struct QueryMetrics {
    std::uint64_t query_ordinal = 0;
    std::string   method;
    std::string   substrate;
    std::string   status;
    std::string   stop_reason;
    // Why the residual was read in full, if it was: "none", when no
    // exactification happened; "cheaper_to_exactify", when the next sampling
    // round would have read more than half the remaining residual; or
    // "gave_up", when planning stopped making progress or the round budget
    // was spent. Recorded after the fact for analysis; it does not steer the
    // control flow.
    std::string   exactify_cause = "none";
    double latency_ms        = 0.0;
    std::uint64_t rows_examined    = 0;
    std::uint64_t measure_reads    = 0;
    std::uint64_t sampled_rows     = 0;
    std::uint64_t exactified_rows  = 0;
    std::uint64_t partitions_touched = 0;
    std::uint64_t partitions_split   = 0;
    std::uint64_t exact_contributors = 0;
    std::uint64_t reusable_strata    = 0;
    std::uint64_t query_local_strata = 0;
    std::uint64_t query_local_exact_contributors = 0;
    std::uint64_t summary_reuse_hits = 0;
    std::uint64_t adaptive_rounds    = 0;
    bool   target_satisfied = false;
    double pre_exactification_error_bound = 0.0;
    std::uint64_t sampling_seed = 0;
};

struct QueryResult {
    std::vector<AggregateEstimate> aggregates;
    QueryMetrics                   metrics;
};

}  // namespace a3i

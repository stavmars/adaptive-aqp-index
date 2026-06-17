// The accuracy-aware query engine: one loop, configured by behavior.
//
// For each query the engine refines the access path (or merely locates, on a
// non-cracking substrate), decomposes the qualifying objects into exact and
// residual contributors, and then either returns an exact answer or runs the
// confidence-bounded read loop until every aggregate meets its target. The
// loop is a two-phase sampling design: a pilot round first raises every
// residual stratum to a small fixed sample
// (taking tiny strata whole) so each stratum's variance is estimated from
// its own rows; each subsequent round re-solves a Neyman allocation from the
// observed statistics, reading a stratum to completion instead whenever the
// plan would cover most of it. Persisted samples from earlier queries count
// toward the pilot, so a revisited stratum plans from real local statistics
// with no new reads. If the round budget runs out -- or a plan cannot raise
// any target while some aggregate still fails -- the remaining residual is
// read in full, the terminal fallback that guarantees termination with a
// correct answer. Reading a stratum to completion is not a separate
// mechanism: it is a read round whose target is the stratum's whole
// population, so it reads only the rows not yet sampled and folds them into
// the same running summaries, after which the stratum contributes exactly
// with zero variance. Sampled progress on whole-partition strata is kept in
// the partition state store, so when summaries are persisted later
// overlapping queries reuse it; progress on partial-overlap strata is
// query-local and discarded at query end (it described only this rectangle).
//
// A single behavior is captured by two flags: whether to honor the query's
// accuracy target or force an exact answer, and whether to keep reusable
// summaries across queries. Whether the substrate may be cracked is a
// property of the substrate itself, read from its refinement predicate.

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "a3i/access_path/adaptive_access_path.hpp"
#include "a3i/aqp/allocator.hpp"
#include "a3i/aqp/estimator.hpp"
#include "a3i/aqp/partition_state_store.hpp"
#include "a3i/aqp/query_decomposition.hpp"
#include "a3i/aqp/summary.hpp"
#include "a3i/core/query.hpp"
#include "a3i/storage/binary_column_store.hpp"
#include "a3i/storage/index_table.hpp"

namespace a3i {

struct EngineConfig {
    enum class AccuracyMode { PerQuery, ForceExact };
    AccuracyMode accuracy_mode = AccuracyMode::PerQuery;
    bool         persist_summaries = false;
    // Sort each round's gathered row ids ascending before reading the measure
    // columns. The columns are stored by original row id and never permuted, so
    // ascending order caps every page at one fault under eviction and turns the
    // device offset stream monotonic -- a win when a column does not fit in RAM
    // and on rotational media. It costs a per-round sort that buys little when
    // the column is resident and the read is sparse. A global toggle, not a
    // per-query heuristic.
    bool         sort_gather_by_row_id = true;
    AllocatorConfig allocator;
};

class QueryEngine {
public:
    QueryEngine(const BinaryColumnStore& store, IndexTable& table,
                AdaptiveAccessPath& access_path, EngineConfig config);

    /// Build the substrate and, when summaries persist and the substrate
    /// prebuilds its partitions, precompute every partition's exact summary up
    /// front so later fully-contained queries answer with no measure reads.
    /// Idempotent; execute() calls it on the first query, but a caller that
    /// wants the up-front cost timed separately may call it explicitly.
    void initialize();

    /// Answer one query. `query_ordinal` seeds the sampling draws so a run is
    /// reproducible.
    QueryResult execute(const RangeQuery& query, std::uint64_t query_ordinal);

    const PartitionStateStore& state_store() const noexcept { return state_; }

private:
    // Per-query description of a residual partition (the sampling unit: all of
    // a partition's measures are drawn over the same rows).
    struct ResidualPartition {
        bool          reusable = false;   // cursor geometry: contiguous range
        bool          write_to_state = false;  // route moments to state_ (persist) vs ql_
        PartitionId   pid = 0;
        IndexPos      begin = 0;
        std::uint32_t size = 0;
        const PositionBitset* qualifying = nullptr;  // query-local only
        std::uint64_t N = 0;
        std::shared_ptr<SampleTracker> tracker;
    };

    // Per-query state of a query-local partition (not persisted).
    struct QueryLocalState {
        std::uint64_t sampled = 0;
        std::vector<MomentStats> moments;  // per measure
        std::shared_ptr<SampleTracker> tracker;
    };

    void build_residual_partitions(const QueryDecomposition& decomp);
    std::vector<StratumAlloc> assemble_allocation() const;
    std::vector<std::vector<StratumSample>> assemble_estimator_input() const;
    // Read one round. For each residual stratum read the rows needed to reach
    // its cumulative `targets[h]` (those not yet sampled), gather every
    // measure over the merged ascending stream, and fold the results into the
    // partition's running summary (persistent for reusable strata, query-local
    // otherwise). Rows read toward a full-population target are charged to
    // the exactified-rows counter, all others to the sampled-rows counter.
    //
    // Scan-to-exactify: if the planned batch's scattered read would instead
    // take the storage scan path (it would sweep the whole [min,max] span
    // anyway), the round escalates to reading the entire residual in that one
    // pass and returns true; the caller then treats the query as exactified.
    // Returns false for an ordinary sampling round. A round already targeting
    // the full residual (the terminal exactify) never escalates.
    bool read_round(const std::vector<std::uint64_t>& targets,
                    std::uint64_t ordinal, std::uint64_t round,
                    QueryMetrics& metrics);
    // Read every residual stratum to completion (target == population), so the
    // remaining rows are read once and the residual variance collapses to zero.
    void exactify_round(std::uint64_t ordinal, std::uint64_t round,
                        QueryMetrics& metrics);
    StratumSample sample_for(const ResidualPartition& p, MeasureId mid) const;
    std::uint64_t sampled_count(const ResidualPartition& p) const;
    bool all_satisfied(const std::vector<AggregateEstimate>& est,
                       double rel) const;

    const BinaryColumnStore& store_;
    IndexTable&              table_;
    AdaptiveAccessPath&      access_path_;
    EngineConfig             config_;
    std::size_t              measure_count_;

    // Behavior-and-substrate derived flags, resolved once at construction.
    bool                     allow_refine_ = false;
    bool                     eager_materialize_ = false;

    bool                initialized_ = false;
    PartitionStateStore state_;
    Estimator           estimator_;
    Allocator           allocator_;

    // Largest absolute per-measure global mean, from the manifest statistics:
    // the scale behind the magnitude floors that keep relative half-widths
    // and variance budgets meaningful near zero totals.
    double global_mean_abs_ = 0.0;

    // Reset per query.
    std::vector<ResidualPartition> residual_;
    std::unordered_map<PartitionId, QueryLocalState> ql_;
};

}  // namespace a3i

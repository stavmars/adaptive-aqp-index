#include "a3i/aqp/decompose.hpp"

#include <memory>
#include <vector>

namespace a3i {

namespace {

// Carries the descent's shared context and accumulates the result so the
// recursion stays a small method instead of a closure with captured refs.
struct Descender {
    const HyperRect&     q;
    AdaptiveAccessPath&  ap;
    PartitionStateStore& store;
    IndexTable&          table;
    std::size_t          measure_count;
    bool                 persist;
    bool                 allow_refine;
    DecompositionResult  result;

    // A contained node already exact for every measure: contribute its one
    // stored summary for its whole contiguous range and stop (no descent into
    // the covered sub-tree).
    void emit_exact(PartitionId pid, bool retained_ancestor,
                    std::uint64_t population) {
        for (std::size_t mid = 0; mid < measure_count; ++mid) {
            const auto m = static_cast<MeasureId>(mid);
            const MeasureSummary* s = store.find(pid, m);
            result.decomposition.exact_contributors.push_back(
                {pid, m, s, retained_ancestor});
            // A complete summary contributes its sampled body moments plus any
            // banked held-out rows (zero on a fully materialized summary, where
            // the held-out rows are already inside non_nan).
            result.exact_bucket.sum_by_measure[mid] +=
                s->non_nan.sum() + s->outlier_sum;
            result.exact_bucket.count_by_measure[mid] +=
                s->non_nan.non_nan_count + s->outlier_count;
        }
        result.total_count += population;
        ++result.frontier_partitions;
        ++result.exact_contributor_partitions;
    }

    // A contained leaf without a complete summary: sample its whole range.
    void emit_reusable(PartitionId pid, const PartitionView& pv,
                       std::uint64_t population) {
        // Classify by the prior summary state, captured before we (possibly)
        // create a fresh summary below: a partition that an earlier query left
        // partially sampled resumes that sample; one with no stored rows
        // (freshly cracked or never sampled) starts from scratch.
        const MeasureSummary* prior = store.find(pid, 0);
        const bool sampled_before = prior != nullptr && prior->sampled_rows > 0;

        // Hold out this partition's flagged rows from the sampling universe and
        // route them to the add-back; the sample then expands over the body
        // only. The excluded bitset is allocated only when flagged rows exist.
        // Locate this partition's held-out (flagged) rows by slicing the global
        // flag column over its range. They are removed from the sampling
        // universe via `excluded`; their values are contributed exactly -- on
        // the persist path banked into the summary (read once in a round, reused
        // thereafter), on the non-persist path added back per query.
        std::shared_ptr<PositionBitset> excluded;
        std::vector<RowId> held_rows;
        if (table.flags_enabled()) {
            table.for_each_flagged_in_range(
                pv.begin, pv.end, [&](IndexPos pos) {
                    if (!excluded) {
                        excluded = std::make_shared<PositionBitset>(population);
                    }
                    excluded->set(static_cast<IndexPos>(pos - pv.begin));
                    held_rows.push_back(table.row_id(pos));
                });
        }
        const std::uint64_t outlier_rows = excluded ? excluded->count() : 0;
        const std::uint64_t body = population - outlier_rows;
        std::shared_ptr<const PositionBitset> excluded_const = std::move(excluded);

        if (persist) {
            store.ensure_partition(pid, measure_count);
            for (std::size_t mid = 0; mid < measure_count; ++mid) {
                const auto m = static_cast<MeasureId>(mid);
                MeasureSummary& sm = store.get_or_create(pid, m, population);
                sm.outlier_rows = outlier_rows;
                // If a prior query already banked these rows, contribute them
                // now from the bank (no read). The round-fold only materializes
                // when outliers_materialized is still false, so the bank-now and
                // bank-in-round paths are mutually exclusive within a query.
                if (sm.outliers_materialized) {
                    result.exact_bucket.sum_by_measure[mid] += sm.outlier_sum;
                    result.exact_bucket.count_by_measure[mid] += sm.outlier_count;
                }
                result.decomposition.reusable_strata.push_back(
                    {pid, m, pv.begin, pv.end, body, sm.tracker, excluded_const});
            }
        } else {
            // No persistent state to bank into: one fresh per-query tracker
            // shared by the partition's measures, and the held-out rows are
            // contributed per query via the add-back.
            for (RowId r : held_rows) result.addback_rows.push_back(r);
            auto tracker = std::make_shared<SampleTracker>(population);
            for (std::size_t mid = 0; mid < measure_count; ++mid) {
                result.decomposition.reusable_strata.push_back(
                    {pid, static_cast<MeasureId>(mid), pv.begin, pv.end,
                     body, tracker, excluded_const});
            }
        }
        result.total_count += population;
        ++result.frontier_partitions;
        if (sampled_before) ++result.reusable_sampled_partitions;
        else                ++result.reusable_absent_partitions;
    }

    // A partial leaf: only the qualifying positions take part; the sample is
    // query-local and its tracker is discarded when the query ends.
    void emit_query_local(PartitionId pid, const PartitionView& pv,
                          std::uint64_t population) {
        auto qualifying = std::make_shared<PositionBitset>(population);
        const bool flags = table.flags_enabled();
        std::uint64_t excluded_qualifying = 0;
        for (std::uint64_t p = 0; p < population; ++p) {
            const auto pos = static_cast<IndexPos>(pv.begin + p);
            if (!q.contains_point(table.point(pos))) continue;
            // A qualifying flagged row is held out of the sample and contributed
            // exactly via the add-back; it still counts toward COUNT(*).
            if (flags && table.is_flagged(pos)) {
                result.addback_rows.push_back(table.row_id(pos));
                ++excluded_qualifying;
            } else {
                qualifying->set(static_cast<IndexPos>(p));
            }
        }
        const std::uint64_t qualifying_count = qualifying->count();
        std::shared_ptr<const PositionBitset> shared = std::move(qualifying);
        for (std::size_t mid = 0; mid < measure_count; ++mid) {
            result.decomposition.query_local_strata.push_back(
                {pid, static_cast<MeasureId>(mid), pv.begin, shared,
                 qualifying_count});
        }
        result.total_count += qualifying_count + excluded_qualifying;
        ++result.frontier_partitions;
        ++result.query_local_partitions;
    }

    void descend(PartitionId pid) {
        ++result.nodes_visited;
        const Containment c = ap.classify(pid, q);
        if (c == Containment::Disjoint) return;

        const PartitionView pv = ap.partition(pid);
        const auto population = static_cast<std::uint64_t>(pv.end - pv.begin);

        if (c == Containment::Contained) {
            // Stop at the highest contained node already exact for every
            // measure; its summary answers the whole sub-tree it covers.
            if (persist && store.is_complete(pid, measure_count)) {
                emit_exact(pid, /*retained_ancestor=*/!ap.is_leaf(pid),
                           population);
                return;
            }
            if (ap.is_leaf(pid)) {
                emit_reusable(pid, pv, population);
                return;
            }
            for (PartitionId ch : ap.overlapping_children(pid, q)) descend(ch);
            return;
        }

        // Partial. A partial leaf is cracked toward the query when the
        // substrate refines, then its children are descended; if it stays a
        // leaf (no refinement, below the crack threshold, or a failed crack)
        // it is scanned into a query-local stratum. A partial interior node is
        // simply descended.
        if (ap.is_leaf(pid)) {
            if (allow_refine) {
                const std::vector<PartitionId> retired =
                    ap.refine(pid, q, table);
                if (persist) {
                    for (PartitionId rid : retired) {
                        // A parent only has stored summaries if it was sampled
                        // before being split; make room so it can be retired
                        // either way, keeping any frozen summary.
                        store.ensure_partition(rid, measure_count);
                        store.retire_partition(rid);
                    }
                }
                // Count this boundary leaf once if cracking actually split it;
                // a refine that produced no cut (below threshold, or a cut
                // plane outside the data) leaves it a leaf to be scanned below.
                if (!retired.empty()) ++result.partitions_refined;
                if (!ap.is_leaf(pid)) {
                    for (PartitionId ch : ap.overlapping_children(pid, q)) descend(ch);
                    return;
                }
            }
            emit_query_local(pid, pv, population);
            return;
        }
        for (PartitionId ch : ap.overlapping_children(pid, q)) descend(ch);
    }
};

}  // namespace

DecompositionResult decompose_descent(const HyperRect& q,
                                      AdaptiveAccessPath& access_path,
                                      PartitionStateStore& state_store,
                                      IndexTable& table,
                                      std::size_t measure_count,
                                      bool persist,
                                      bool allow_refine) {
    Descender d{q,       access_path,  state_store, table, measure_count,
                persist, allow_refine, {}};
    d.result.exact_bucket.sum_by_measure.assign(measure_count, 0.0);
    d.result.exact_bucket.count_by_measure.assign(measure_count, 0);

    for (PartitionId root : access_path.roots()) d.descend(root);

    // get_or_create() for reusable strata may have grown the store's backing
    // vector after an exact contributor took a summary pointer, invalidating
    // it. Re-resolve every exact contributor's pointer now that the store has
    // stopped growing.
    for (ExactContributor& c : d.result.decomposition.exact_contributors) {
        c.summary = state_store.find(c.pid, c.mid);
    }
    return d.result;
}

}  // namespace a3i

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
            result.exact_bucket.sum_by_measure[mid] += s->non_nan.sum();
            result.exact_bucket.count_by_measure[mid] += s->non_nan.non_nan_count;
        }
        result.total_count += population;
        ++result.partitions_touched;
    }

    // A contained leaf without a complete summary: sample its whole range.
    void emit_reusable(PartitionId pid, const PartitionView& pv,
                       std::uint64_t population) {
        if (persist) {
            store.ensure_partition(pid, measure_count);
            for (std::size_t mid = 0; mid < measure_count; ++mid) {
                const auto m = static_cast<MeasureId>(mid);
                MeasureSummary& sm = store.get_or_create(pid, m, population);
                result.decomposition.reusable_strata.push_back(
                    {pid, m, pv.begin, pv.end, population, sm.tracker});
            }
        } else {
            // No persistence: one fresh per-query tracker shared by the
            // partition's measures, never stored.
            auto tracker = std::make_shared<SampleTracker>(population);
            for (std::size_t mid = 0; mid < measure_count; ++mid) {
                result.decomposition.reusable_strata.push_back(
                    {pid, static_cast<MeasureId>(mid), pv.begin, pv.end,
                     population, tracker});
            }
        }
        result.total_count += population;
        ++result.partitions_touched;
    }

    // A partial leaf: only the qualifying positions take part; the sample is
    // query-local and its tracker is discarded when the query ends.
    void emit_query_local(PartitionId pid, const PartitionView& pv,
                          std::uint64_t population) {
        auto qualifying = std::make_shared<PositionBitset>(population);
        for (std::uint64_t p = 0; p < population; ++p) {
            const auto pos = static_cast<IndexPos>(pv.begin + p);
            if (q.contains_point(table.point(pos))) {
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
        result.total_count += qualifying_count;
        ++result.partitions_touched;
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
            for (PartitionId ch : ap.children(pid)) descend(ch);
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
                result.partitions_split += retired.size();
                if (!ap.is_leaf(pid)) {
                    for (PartitionId ch : ap.children(pid)) descend(ch);
                    return;
                }
            }
            emit_query_local(pid, pv, population);
            return;
        }
        for (PartitionId ch : ap.children(pid)) descend(ch);
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

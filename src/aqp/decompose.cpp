#include "a3i/aqp/decompose.hpp"

namespace a3i {

DecompositionResult decompose(const HyperRect& q,
                              const AdaptiveAccessPath& access_path,
                              const QueryPartitionSet& frontier,
                              PartitionStateStore& state_store,
                              const IndexTable& table,
                              std::size_t measure_count) {
    DecompositionResult result;
    result.exact_bucket.sum_by_measure.assign(measure_count, 0.0);
    result.exact_bucket.count_by_measure.assign(measure_count, 0);

    // Make room in the store for every frontier partition up front. Later
    // get_or_create() calls then only fill existing slots, so the summary
    // pointers taken for exact contributors stay valid for the whole pass.
    for (PartitionId pid : frontier.fully_contained) {
        state_store.ensure_partition(pid, measure_count);
    }
    for (PartitionId pid : frontier.partial) {
        state_store.ensure_partition(pid, measure_count);
    }

    for (PartitionId pid : frontier.fully_contained) {
        const PartitionView pv = access_path.partition(pid);
        const std::uint64_t population =
            static_cast<std::uint64_t>(pv.end - pv.begin);

        bool all_exact = true;
        for (std::size_t mid = 0; mid < measure_count; ++mid) {
            const MeasureSummary* s =
                state_store.find(pid, static_cast<MeasureId>(mid));
            if (s == nullptr || !s->complete()) {
                all_exact = false;
                break;
            }
        }

        if (all_exact) {
            for (std::size_t mid = 0; mid < measure_count; ++mid) {
                const auto m = static_cast<MeasureId>(mid);
                const MeasureSummary* s = state_store.find(pid, m);
                result.decomposition.exact_contributors.push_back(
                    {pid, m, s, /*retained_ancestor=*/false});
                result.exact_bucket.sum_by_measure[mid] += s->non_nan.sum();
                result.exact_bucket.count_by_measure[mid] +=
                    s->non_nan.non_nan_count;
            }
        } else {
            for (std::size_t mid = 0; mid < measure_count; ++mid) {
                const auto m = static_cast<MeasureId>(mid);
                MeasureSummary& sm = state_store.get_or_create(pid, m, population);
                result.decomposition.reusable_strata.push_back(
                    {pid, m, pv.begin, pv.end, population, sm.tracker});
            }
        }
        result.total_count += population;
    }

    for (PartitionId pid : frontier.partial) {
        const PartitionView pv = access_path.partition(pid);
        const std::uint64_t population =
            static_cast<std::uint64_t>(pv.end - pv.begin);

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
    }

    return result;
}

}  // namespace a3i

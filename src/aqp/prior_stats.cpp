#include "a3i/aqp/prior_stats.hpp"

#include <stdexcept>
#include <utility>

namespace a3i {

PriorStatsProvider::PriorStatsProvider(
    const PartitionStateStore& store, ParentLookup parent_of,
    const std::vector<GlobalMeasureStats>& global_stats)
    : store_(store),
      parent_of_(std::move(parent_of)),
      global_stats_(global_stats) {}

const MeasureSummary* PriorStatsProvider::complete_partition_summary(
    PartitionId id, MeasureId mid) const {
    const MeasureSummary* s = store_.find(id, mid);
    return (s && s->complete()) ? s : nullptr;
}

const MeasureSummary* PriorStatsProvider::nearest_retained_exact_ancestor(
    PartitionId id, MeasureId mid) const {
    std::optional<PartitionId> cur = parent_of_(id);
    while (cur) {
        const MeasureSummary* s = store_.find(*cur, mid);
        if (s && s->complete()) return s;
        cur = parent_of_(*cur);
    }
    return nullptr;
}

const GlobalMeasureStats& PriorStatsProvider::global_measure_stats(
    MeasureId mid) const {
    if (mid >= global_stats_.size()) {
        throw std::out_of_range("PriorStatsProvider::global_measure_stats unknown measure");
    }
    return global_stats_[mid];
}

}  // namespace a3i

#include "a3i/aqp/partition_state_store.hpp"

#include <stdexcept>
#include <utility>

namespace a3i {

PartitionId PartitionStateStore::register_partition(std::size_t measure_count) {
    const auto id = static_cast<PartitionId>(states_by_partition_.size());
    PartitionRuntimeState st;
    st.current = true;
    st.summaries_by_measure.resize(measure_count);
    states_by_partition_.push_back(std::move(st));
    return id;
}

void PartitionStateStore::ensure_partition(PartitionId id,
                                           std::size_t measure_count) {
    if (id >= states_by_partition_.size()) {
        states_by_partition_.resize(static_cast<std::size_t>(id) + 1);
    }
    auto& st = states_by_partition_[id];
    if (st.summaries_by_measure.size() < measure_count) {
        st.summaries_by_measure.resize(measure_count);
    }
}

void PartitionStateStore::retire_partition(PartitionId id) {
    state(id).current = false;
}

bool PartitionStateStore::is_current(PartitionId id) const {
    return state(id).current;
}

const MeasureSummary* PartitionStateStore::find(PartitionId id,
                                                MeasureId mid) const {
    if (id >= states_by_partition_.size()) return nullptr;
    const auto& summaries = states_by_partition_[id].summaries_by_measure;
    if (mid >= summaries.size()) return nullptr;
    const auto& slot = summaries[mid];
    return slot ? &*slot : nullptr;
}

bool PartitionStateStore::is_complete(PartitionId id,
                                      std::size_t measure_count) const {
    if (id >= states_by_partition_.size()) return false;
    const auto& summaries = states_by_partition_[id].summaries_by_measure;
    if (summaries.size() < measure_count) return false;
    for (std::size_t mid = 0; mid < measure_count; ++mid) {
        const auto& slot = summaries[mid];
        if (!slot || !slot->complete()) return false;
    }
    return true;
}

MeasureSummary& PartitionStateStore::get_or_create(
    PartitionId id, MeasureId mid, std::uint64_t population_size) {
    auto& st = state(id);
    if (mid >= st.summaries_by_measure.size()) {
        throw std::out_of_range("PartitionStateStore::get_or_create unknown measure");
    }
    if (st.tracker == nullptr) {
        st.tracker = std::make_shared<SampleTracker>(population_size);
    }
    auto& slot = st.summaries_by_measure[mid];
    if (!slot) {
        MeasureSummary s;
        s.population_size = population_size;
        s.tracker = st.tracker;
        slot = std::move(s);
    }
    return *slot;
}

void PartitionStateStore::update_sampled(PartitionId id, MeasureId mid,
                                         const SampleDelta& delta) {
    auto& st = state(id);
    if (mid >= st.summaries_by_measure.size() || !st.summaries_by_measure[mid]) {
        throw std::logic_error("PartitionStateStore::update_sampled missing summary");
    }
    MeasureSummary& s = *st.summaries_by_measure[mid];
    s.sampled_rows += delta.new_sampled_rows;
    s.non_nan.merge(delta.moments);
}

void PartitionStateStore::replace_with_complete(PartitionId id, MeasureId mid,
                                                MeasureSummary summary) {
    auto& st = state(id);
    if (mid >= st.summaries_by_measure.size()) {
        throw std::out_of_range("PartitionStateStore::replace_with_complete unknown measure");
    }
    st.summaries_by_measure[mid] = std::move(summary);
}

PartitionRuntimeState& PartitionStateStore::state(PartitionId id) {
    if (id >= states_by_partition_.size()) {
        throw std::out_of_range("PartitionStateStore: unknown partition id");
    }
    return states_by_partition_[id];
}

const PartitionRuntimeState& PartitionStateStore::state(PartitionId id) const {
    if (id >= states_by_partition_.size()) {
        throw std::out_of_range("PartitionStateStore: unknown partition id");
    }
    return states_by_partition_[id];
}

}  // namespace a3i

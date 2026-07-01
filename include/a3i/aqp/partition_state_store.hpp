// Dense, O(1) store of per-partition aggregate summaries.
//
// Partition ids are dense and consecutive, so they index a vector directly
// with no hashing. Each partition owns a small vector of optional summaries
// indexed by MeasureId, plus one SampleTracker shared by all of its measures
// (a partition's measures are sampled in lockstep over the same drawn rows).
// Splitting a partition retires it -- its summaries are kept as valid
// summaries of the parent's object set -- while children start absent.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "a3i/aqp/summary.hpp"
#include "a3i/core/types.hpp"

namespace a3i {

/// Per-partition slot: whether the partition is a live leaf, the summaries
/// for its measures, and the tracker shared across those measures.
struct PartitionRuntimeState {
    bool current = true;
    std::vector<std::optional<MeasureSummary>> summaries_by_measure;
    std::shared_ptr<SampleTracker> tracker;  // created with the first summary
};

class PartitionStateStore {
public:
    /// Append a new partition with room for `measure_count` measures and
    /// return its dense id.
    PartitionId register_partition(std::size_t measure_count);

    /// Make sure `id` exists and has room for `measure_count` measures,
    /// growing the backing storage as needed.
    void ensure_partition(PartitionId id, std::size_t measure_count);

    /// Mark a partition non-current after a split; its summaries are kept.
    void retire_partition(PartitionId id);

    /// True iff the partition is a live leaf (not retired).
    bool is_current(PartitionId id) const;

    /// Summary for the pair, or nullptr if absent (or the partition/measure
    /// is unknown).
    const MeasureSummary* find(PartitionId id, MeasureId mid) const;

    /// True iff `id` carries a complete (exact) summary for every one of the
    /// first `measure_count` measures. A contained node that is complete can
    /// answer its whole sub-tree from its stored summaries, so the descent
    /// stops there. False for an unknown partition or any absent/partial
    /// summary.
    bool is_complete(PartitionId id, std::size_t measure_count) const;

    /// Summary for the pair, creating an absent one bound to the partition's
    /// shared tracker (sized `population_size`) on first use. `population_size`
    /// must agree across a partition's measures.
    MeasureSummary& get_or_create(PartitionId id, MeasureId mid,
                                  std::uint64_t population_size);

    /// Fold one round of newly sampled rows into an existing summary.
    void update_sampled(PartitionId id, MeasureId mid, const SampleDelta& delta);

    /// Bank a partition's held-out rows' exact contribution for one measure and
    /// mark the partition's outliers materialized, so a later containing query
    /// reuses them without re-reading. Kept separate from the sampled moments.
    void bank_outliers(PartitionId id, MeasureId mid, double sum,
                       std::uint64_t count);

    /// Overwrite a summary with a completed (exact) one.
    void replace_with_complete(PartitionId id, MeasureId mid,
                               MeasureSummary summary);

    std::size_t partition_count() const noexcept {
        return states_by_partition_.size();
    }

private:
    PartitionRuntimeState&       state(PartitionId id);
    const PartitionRuntimeState& state(PartitionId id) const;

    std::vector<PartitionRuntimeState> states_by_partition_;
};

}  // namespace a3i

// Source of prior aggregate statistics used when allocating samples.
//
// Priors are consulted in decreasing order of sharpness:
//   1. the partition's own complete summary (exact, if it exists);
//   2. the nearest retained exact ancestor's summary (a sharper-than-global
//      prior left behind when this partition's ancestor was fully read);
//   3. the global per-measure statistics computed once at conversion.
// Ancestry is resolved through a caller-supplied parent lookup so the
// provider stays independent of any concrete access-path type.

#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "a3i/aqp/partition_state_store.hpp"
#include "a3i/aqp/summary.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/manifest.hpp"

namespace a3i {

class PriorStatsProvider {
public:
    /// Maps a partition to its parent, or nullopt for a root.
    using ParentLookup = std::function<std::optional<PartitionId>(PartitionId)>;

    PriorStatsProvider(const PartitionStateStore& store,
                       ParentLookup parent_of,
                       const std::vector<GlobalMeasureStats>& global_stats);

    /// The partition's own summary iff it is complete (exact); else nullptr.
    const MeasureSummary* complete_partition_summary(PartitionId id,
                                                     MeasureId mid) const;

    /// The closest strict ancestor whose summary for this measure is
    /// complete (exact); nullptr if none up to the root.
    const MeasureSummary* nearest_retained_exact_ancestor(PartitionId id,
                                                          MeasureId mid) const;

    /// Global statistics for the measure (the coarsest prior).
    const GlobalMeasureStats& global_measure_stats(MeasureId mid) const;

private:
    const PartitionStateStore&             store_;
    ParentLookup                           parent_of_;
    const std::vector<GlobalMeasureStats>& global_stats_;
};

}  // namespace a3i

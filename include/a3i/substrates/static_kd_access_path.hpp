// Fully-built, non-adaptive KD-tree substrate (the classic full-index
// baseline).
//
// ensure_built() performs the complete top-down build in one shot: each node
// is split at the median value of a round-robin axis until its population
// drops to leaf_min_size (or the axis is degenerate and cannot be split).
// The whole tree therefore exists before query 0 returns; its construction
// cost is charged to query 0 like any other substrate build.
//
// refine() is a genuine no-op: the structure never adapts. It still returns
// the frontier classification of the query (== locate) so callers share one
// code path with the adaptive substrate.
//
// The median value partition is unstable, so ranges_are_row_id_ordered() is
// false and the read path sorts per stratum, exactly as for the adaptive
// substrate.

#pragma once

#include <optional>
#include <vector>

#include "a3i/access_path/adaptive_access_path.hpp"
#include "a3i/access_path/substrate_factory.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/kd_tree.hpp"

namespace a3i {

class StaticKdAccessPath : public AdaptiveAccessPath {
public:
    explicit StaticKdAccessPath(SubstrateConfig config);

    void prepare(IndexTable& table) override;
    void ensure_built() override;
    QueryPartitionSet locate(const HyperRect& q) const override;
    RefineResult refine(const HyperRect& q, IndexTable& table) override;
    PartitionView partition(PartitionId id) const override;
    std::vector<PartitionId> active_partitions() const override;
    std::optional<PartitionId> parent(PartitionId id) const override;
    bool ranges_are_row_id_ordered() const override { return false; }
    bool supports_refine() const override { return false; }
    bool is_fully_built() const override { return true; }

private:
    // Recursively split node `id` on `axis` at its median value, cycling the
    // axis each level, until its population is at or below leaf_min_size or
    // the split makes no progress (a degenerate, all-equal axis range).
    void build(PartitionId id, DimensionId axis);

    // Median coordinate of node `id`'s range on `axis`, used as the split
    // value: the left child takes points below it, the right child the rest.
    double median_value(PartitionId id, DimensionId axis) const;

    SubstrateConfig config_;
    IndexTable*     table_ = nullptr;  ///< Non-owning; prepared at load time.
    bool            built_ = false;
    KdTree          tree_;
};

}  // namespace a3i

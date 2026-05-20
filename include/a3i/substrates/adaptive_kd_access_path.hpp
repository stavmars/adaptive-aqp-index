// Single-root adaptive KD-tree substrate.
//
// ensure_built() creates one root partition over the whole table [0, N);
// from there the structure grows by query-bound cracking. Each refine()
// splits the boundary partitions that exceed the refinement threshold,
// isolating the query rectangle so that later queries over the same region
// reuse fully-contained children.
//
// Positions are permuted by cracking (an unstable in-place partition), so
// ranges_are_row_id_ordered() is false: the read path sorts per stratum.

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

class AdaptiveKdAccessPath : public AdaptiveAccessPath {
public:
    explicit AdaptiveKdAccessPath(SubstrateConfig config);

    void prepare(IndexTable& table) override;
    void ensure_built() override;
    QueryPartitionSet locate(const HyperRect& q) const override;
    RefineResult refine(const HyperRect& q, IndexTable& table) override;
    PartitionView partition(PartitionId id) const override;
    std::vector<PartitionId> active_partitions() const override;
    std::optional<PartitionId> parent(PartitionId id) const override;
    bool ranges_are_row_id_ordered() const override { return false; }

private:
    // Isolate q from one boundary leaf: crack at each axis lower bound
    // (descending into the >= child) then each upper bound (descending into
    // the < child), stopping as soon as the surviving boundary child is no
    // larger than the refinement threshold. Appends every retired parent id
    // and returns the id of the surviving boundary child (== `id` if no
    // crack succeeded). Every discarded child lies wholly outside q.
    PartitionId crack_partition_to_query(PartitionId id, const HyperRect& q,
                                         std::vector<PartitionId>& retired);

    SubstrateConfig config_;
    IndexTable*     table_ = nullptr;  ///< Non-owning; prepared at load time.
    bool            built_ = false;
    KdTree          tree_;
};

}  // namespace a3i

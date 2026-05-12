// Single-root adaptive KD-tree substrate.
//
// ensure_built() creates one root partition over the whole table [0, N);
// from there the structure grows by query-bound cracking. The node layout
// (contiguous [begin, end), per-node bounds, split axis/pivot, child and
// parent links) is laid out for that cracking even though the splitting
// itself arrives later; for now refine() is a no-op so a freshly built
// path is a single boundary partition that exercises the full query layer.
//
// Positions are permuted by cracking (an unstable in-place partition), so
// ranges_are_row_id_ordered() is false: the read path sorts per stratum.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "a3i/access_path/adaptive_access_path.hpp"
#include "a3i/access_path/substrate_factory.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/index_table.hpp"

namespace a3i {

class AdaptiveKdAccessPath : public AdaptiveAccessPath {
public:
    explicit AdaptiveKdAccessPath(SubstrateConfig config);

    void prepare(IndexTable& table) override;
    void ensure_built() override;
    QueryPartitionSet locate(const HyperRect& q) const override;
    std::vector<PartitionId> refine(const HyperRect& q, IndexTable& table) override;
    PartitionView partition(PartitionId id) const override;
    std::vector<PartitionId> active_partitions() const override;
    std::optional<PartitionId> parent(PartitionId id) const override;
    bool ranges_are_row_id_ordered() const override { return false; }

private:
    struct Node {
        PartitionId id    = 0;
        HyperRect   bounds;
        IndexPos    begin = 0;
        IndexPos    end   = 0;
        bool        leaf  = true;
        bool        active = true;
        // Split metadata, set when the node becomes internal (cracking).
        DimensionId axis  = 0;
        double      pivot = 0.0;
        std::optional<PartitionId> left;
        std::optional<PartitionId> right;
        std::optional<PartitionId> parent;
    };

    // Walk the live structure for `q`, classifying each active leaf.
    void descend(PartitionId id, const HyperRect& q, QueryPartitionSet& out) const;

    SubstrateConfig    config_;
    IndexTable*        table_ = nullptr;   ///< Non-owning; prepared at load time.
    bool               built_ = false;
    std::vector<Node>  nodes_;             ///< Indexed by PartitionId (dense, append-only).
};

}  // namespace a3i

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
#include <utility>
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
    RefineResult refine(const HyperRect& q, IndexTable& table) override;
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

    // In-place Hoare two-pointer partition of positions [start, end) on one
    // axis about `pivot`: points with coordinate < pivot are moved before
    // those >= pivot, each swap carrying its row_id. Returns the first
    // position of the >= group (== start or == end when every point lands
    // on one side, i.e. a failed crack).
    IndexPos hoare_partition(IndexPos start, IndexPos end, DimensionId axis,
                             double pivot);

    // Split leaf `id` about (axis, pivot): partition its range, and on a
    // successful crack create the two leaf children, retire the parent, and
    // return their ids {left (< pivot), right (>= pivot)}. Returns nullopt on
    // a failed crack (the node stays an unchanged leaf).
    std::optional<std::pair<PartitionId, PartitionId>>
    crack_node(PartitionId id, DimensionId axis, double pivot);

    // Isolate q from one boundary leaf: crack at each axis lower bound
    // (descending into the >= child) then each upper bound (descending into
    // the < child), stopping as soon as the surviving boundary child is no
    // larger than the refinement threshold. Appends every retired parent id
    // and returns the id of the surviving boundary child (== `id` if no
    // crack succeeded). Every discarded child lies wholly outside q.
    PartitionId crack_partition_to_query(PartitionId id, const HyperRect& q,
                                         std::vector<PartitionId>& retired);

    SubstrateConfig    config_;
    IndexTable*        table_ = nullptr;   ///< Non-owning; prepared at load time.
    bool               built_ = false;
    std::vector<Node>  nodes_;             ///< Indexed by PartitionId (dense, append-only).
};

}  // namespace a3i

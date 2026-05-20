// Shared binary KD partitioning over an index table's positions.
//
// A node owns a contiguous [begin, end) slice of the table and a spatial
// bounding rectangle. An internal node splits one axis at a pivot value:
// the left child owns the points whose coordinate on that axis is < pivot,
// the right child owns those >= pivot, and their bounds tile the parent's
// exactly under the half-open [low, high) convention. Leaves are the live
// partitions; retired parents are kept for ancestry.
//
// Two substrates grow this same structure and answer locate/partition/
// parent/active identically. They differ only in which pivots they choose
// and when: one splits incrementally on query bounds, the other splits once
// on median values at build time. The choice-and-timing logic lives in the
// substrates; the node layout, the in-place value partition, and the read
// traversal live here.

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "a3i/access_path/partition_view.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/index_table.hpp"

namespace a3i {

class KdTree {
public:
    struct Node {
        PartitionId id     = 0;
        HyperRect   bounds;
        IndexPos    begin  = 0;
        IndexPos    end    = 0;
        bool        leaf   = true;
        bool        active = true;
        // Split metadata, set when the node becomes internal.
        DimensionId axis  = 0;
        double      pivot = 0.0;
        std::optional<PartitionId> left;
        std::optional<PartitionId> right;
        std::optional<PartitionId> parent;
    };

    /// Discard any existing structure and install a single root leaf owning
    /// [0, n) with the given bounds.
    void reset(const HyperRect& root_bounds, IndexPos n);

    bool empty() const { return nodes_.empty(); }

    /// Classify every active leaf against `q`: fully-contained or partial.
    /// Disjoint leaves appear in neither list.
    QueryPartitionSet locate(const HyperRect& q) const;

    /// Read-only view of a node, valid for retired ids too (active = false).
    PartitionView partition(PartitionId id) const;

    /// Ids of the live (active leaf) partitions.
    std::vector<PartitionId> active_partitions() const;

    /// Parent id of a node, or nullopt for the root.
    std::optional<PartitionId> parent(PartitionId id) const;

    const Node& node(PartitionId id) const { return nodes_[id]; }

    /// Population of a node's range, a constant-time size.
    IndexPos population(PartitionId id) const {
        return nodes_[id].end - nodes_[id].begin;
    }

    /// Split leaf `id` about (axis, pivot) by partitioning its range in
    /// `table` in place (each swap carries its row_id). On a successful split
    /// create the two leaf children (left owns < pivot, right owns >= pivot),
    /// retire the parent, and return {left_id, right_id}. Returns nullopt
    /// when every point falls on one side (no progress): the node is left an
    /// unchanged leaf.
    std::optional<std::pair<PartitionId, PartitionId>>
    split_node(IndexTable& table, PartitionId id, DimensionId axis,
               double pivot);

private:
    // In-place Hoare two-pointer partition of positions [start, end) on one
    // axis about `pivot`: points with coordinate < pivot are moved before
    // those >= pivot. Returns the first position of the >= group (== start or
    // == end when every point lands on one side).
    IndexPos partition_range(IndexTable& table, IndexPos start, IndexPos end,
                             DimensionId axis, double pivot) const;

    void descend(PartitionId id, const HyperRect& q,
                 QueryPartitionSet& out) const;

    std::vector<Node> nodes_;  ///< Indexed by PartitionId (dense, append-only).
};

}  // namespace a3i

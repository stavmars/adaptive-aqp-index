// Shared binary KD partitioning over an index table's positions.
//
// A node owns a contiguous [begin, end) slice of the table and a spatial
// bounding rectangle. An internal node splits one axis at a pivot value:
// the left child owns the points whose coordinate on that axis is < pivot,
// the right child owns those >= pivot, and their bounds tile the parent's
// exactly under the half-open [low, high) convention. Leaves are the live
// partitions; retired parents are kept for ancestry.
//
// Two substrates grow this same structure and answer classify/children/
// partition/parent/active identically. They differ only in which pivots they
// choose and when: one splits incrementally on query bounds, the other splits
// once on median values at build time. The choice-and-timing logic lives in
// the substrates; the node layout, the in-place value partition, and the
// navigation primitives live here.

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

    /// Discard any existing structure and install a root (id 0) over [0, n)
    /// holding a pre-built flat set of child partitions. The root is an inactive
    /// parent whose children are addressed by the caller rather than through the
    /// binary left/right links; each child is an active leaf owning the
    /// contiguous range [offsets[i], offsets[i+1]) with bounds child_bounds[i],
    /// and may later be split by split_node(). `offsets` holds
    /// child_bounds.size()+1 ascending entries, starting at 0 and ending at n.
    void reset_with_children(const HyperRect& root_bounds, IndexPos n,
                             const std::vector<HyperRect>& child_bounds,
                             const std::vector<IndexPos>& offsets);

    /// Root bounds for an index over `table` whose data extent is `data_bounds`
    /// (per-axis [min, max]; pass an empty rect to derive it by scanning).
    ///
    /// Partition bounds tile half-open [low, high): the upper edge is
    /// exclusive, so the root's top must sit strictly above the largest value,
    /// otherwise a point on the maximum would land on the rightmost partition's
    /// excluded top edge and be miscounted. Each axis' upper bound is therefore
    /// lifted one representable step past the data (std::nextafter); the lower
    /// bound, which is inclusive, stays at the data minimum.
    ///
    /// The root is sized to the data, not to an infinite or externally declared
    /// domain, on purpose. A finite top matching the data lets a query that
    /// spans the whole range be recognised as fully containing the root and
    /// answered from precomputed summaries with zero row reads; an unbounded
    /// top could never compare as "contained" for any finite query, forcing the
    /// widest queries to re-read the data they already cover.
    static HyperRect compute_root_bounds(const HyperRect& data_bounds,
                                         const IndexTable& table);

    bool empty() const { return nodes_.empty(); }

    /// Entry points of a descent: the single root id {0} when non-empty.
    std::vector<PartitionId> roots() const;

    /// Children of a node, left then right; empty for a leaf.
    std::vector<PartitionId> children(PartitionId id) const;

    /// True iff `id` is a leaf (no children).
    bool is_leaf(PartitionId id) const;

    /// Classify one node's bounds against `q`: disjoint, fully contained, or
    /// partially overlapping. Pure geometry, valid for any node id.
    Containment classify(PartitionId id, const HyperRect& q) const;

    /// Read-only view of a node, valid for retired ids too (active = false).
    PartitionView partition(PartitionId id) const;

    /// Ids of the live (active leaf) partitions. Scans every node and
    /// allocates a fresh vector, so its cost grows with the total node count
    /// (live + retired). Intended for enumeration and tests; do NOT call it
    /// on a per-query path.
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

    std::vector<Node> nodes_;  ///< Indexed by PartitionId (dense, append-only).
};

}  // namespace a3i

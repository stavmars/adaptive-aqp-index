#include "a3i/substrates/kd_tree.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace a3i {

void KdTree::reset(const HyperRect& root_bounds, IndexPos n) {
    nodes_.clear();
    Node root;
    root.id     = 0;
    root.bounds = root_bounds;
    root.begin  = 0;
    root.end    = n;
    root.leaf   = true;
    root.active = true;
    nodes_.push_back(std::move(root));
}

void KdTree::reset_with_children(const HyperRect& root_bounds, IndexPos n,
                                 const std::vector<HyperRect>& child_bounds,
                                 const std::vector<IndexPos>& offsets) {
    if (offsets.size() != child_bounds.size() + 1) {
        throw std::invalid_argument(
            "KdTree::reset_with_children: offsets size must be child count + 1");
    }
    if (offsets.front() != 0 || offsets.back() != n) {
        throw std::invalid_argument(
            "KdTree::reset_with_children: offsets must start at 0 and end at n");
    }

    nodes_.clear();
    nodes_.reserve(child_bounds.size() + 1);

    // Node 0 is the parent over the whole range; it is inactive and its
    // children are addressed by the caller, so the binary left/right links
    // stay unset.
    Node root;
    root.id     = 0;
    root.bounds = root_bounds;
    root.begin  = 0;
    root.end    = n;
    root.leaf   = false;
    root.active = false;
    nodes_.push_back(std::move(root));

    for (std::size_t i = 0; i < child_bounds.size(); ++i) {
        if (offsets[i + 1] < offsets[i]) {
            throw std::invalid_argument(
                "KdTree::reset_with_children: offsets must be ascending");
        }
        Node child;
        child.id     = static_cast<PartitionId>(i + 1);
        child.bounds = child_bounds[i];
        child.begin  = offsets[i];
        child.end    = offsets[i + 1];
        child.leaf   = true;
        child.active = true;
        child.parent = 0;
        nodes_.push_back(std::move(child));
    }
}

HyperRect KdTree::compute_root_bounds(const HyperRect& data_bounds,
                                      const IndexTable& table) {
    const std::size_t d = table.dimensions();
    std::vector<double> lo(d, std::numeric_limits<double>::infinity());
    std::vector<double> hi(d, -std::numeric_limits<double>::infinity());
    if (data_bounds.dims.size() == d) {
        // Extent supplied (e.g. from the manifest): no data scan needed.
        for (std::size_t i = 0; i < d; ++i) {
            lo[i] = data_bounds.dims[i].low;
            hi[i] = data_bounds.dims[i].high;
        }
    } else {
        // Fall back to a one-time scan when the extent is unknown.
        const IndexPos n = static_cast<IndexPos>(table.size());
        for (IndexPos p = 0; p < n; ++p) {
            for (std::size_t i = 0; i < d; ++i) {
                const double v = table.dim(p, static_cast<DimensionId>(i));
                if (v < lo[i]) lo[i] = v;
                if (v > hi[i]) hi[i] = v;
            }
        }
    }
    HyperRect root;
    root.dims.reserve(d);
    for (std::size_t i = 0; i < d; ++i) {
        if (hi[i] < lo[i]) {
            // No data on this axis: a degenerate but valid range.
            root.dims.push_back(Range{0.0, 0.0});
            continue;
        }
        // Lift the exclusive top one representable step past the data so no
        // point sits on a partition's excluded upper edge.
        root.dims.push_back(Range{
            lo[i], std::nextafter(hi[i], std::numeric_limits<double>::infinity())});
    }
    return root;
}

std::vector<PartitionId> KdTree::roots() const {
    if (nodes_.empty()) return {};
    return {0};
}

std::vector<PartitionId> KdTree::children(PartitionId id) const {
    const Node& node = nodes_[id];
    if (node.leaf) return {};
    std::vector<PartitionId> out;
    if (node.left)  out.push_back(*node.left);
    if (node.right) out.push_back(*node.right);
    return out;
}

bool KdTree::is_leaf(PartitionId id) const { return nodes_[id].leaf; }

Containment KdTree::classify(PartitionId id, const HyperRect& q) const {
    const Node& node = nodes_[id];
    if (!node.bounds.intersects(q)) return Containment::Disjoint;
    if (q.contains_rect(node.bounds)) return Containment::Contained;
    return Containment::Partial;
}

IndexPos KdTree::partition_range(IndexTable& table, IndexPos start,
                                 IndexPos end, DimensionId axis,
                                 double pivot) const {
    // Signed cursors so the right cursor can fall below `start` without
    // unsigned wraparound when every point belongs to the >= group.
    std::int64_t i = static_cast<std::int64_t>(start);
    std::int64_t j = static_cast<std::int64_t>(end) - 1;
    while (i <= j) {
        if (table.dim(static_cast<IndexPos>(i), axis) < pivot) {
            ++i;
        } else {
            while (j >= i && table.dim(static_cast<IndexPos>(j), axis) >= pivot) {
                --j;
            }
            if (i < j) {
                table.swap_positions(static_cast<IndexPos>(i),
                                     static_cast<IndexPos>(j));
                ++i;
                --j;
            }
        }
    }
    return static_cast<IndexPos>(i);
}

std::optional<std::pair<PartitionId, PartitionId>>
KdTree::split_node(IndexTable& table, PartitionId id, DimensionId axis,
                   double pivot) {
    const IndexPos begin = nodes_[id].begin;
    const IndexPos end   = nodes_[id].end;
    const IndexPos split = partition_range(table, begin, end, axis, pivot);
    if (split == begin || split == end) {
        return std::nullopt;  // no progress: every point fell on one side
    }

    const PartitionId left_id  = static_cast<PartitionId>(nodes_.size());
    const PartitionId right_id = left_id + 1;

    // Children inherit the parent's bounds, clipped at the pivot on `axis`
    // (half-open: left owns [low, pivot), right owns [pivot, high)).
    Node left;
    left.id                      = left_id;
    left.bounds                  = nodes_[id].bounds;
    left.bounds.dims[axis].high  = pivot;
    left.begin                   = begin;
    left.end                     = split;
    left.leaf                    = true;
    left.active                  = true;
    left.parent                  = id;

    Node right;
    right.id                     = right_id;
    right.bounds                 = nodes_[id].bounds;
    right.bounds.dims[axis].low  = pivot;
    right.begin                  = split;
    right.end                    = end;
    right.leaf                   = true;
    right.active                 = true;
    right.parent                 = id;

    nodes_.push_back(std::move(left));
    nodes_.push_back(std::move(right));

    // Re-index after the appends, which may have reallocated the vector.
    Node& parent  = nodes_[id];
    parent.leaf   = false;
    parent.active = false;
    parent.axis   = axis;
    parent.pivot  = pivot;
    parent.left   = left_id;
    parent.right  = right_id;
    return std::make_pair(left_id, right_id);
}

PartitionView KdTree::partition(PartitionId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("KdTree::partition unknown id");
    }
    const Node& node = nodes_[id];
    return PartitionView{node.id, node.bounds, node.begin, node.end, node.active};
}

std::vector<PartitionId> KdTree::active_partitions() const {
    std::vector<PartitionId> ids;
    for (const Node& node : nodes_) {
        if (node.active && node.leaf) ids.push_back(node.id);
    }
    return ids;
}

std::optional<PartitionId> KdTree::parent(PartitionId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("KdTree::parent unknown id");
    }
    return nodes_[id].parent;
}

}  // namespace a3i

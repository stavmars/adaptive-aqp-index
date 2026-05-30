#include "a3i/substrates/kd_tree.hpp"

#include <cstdint>
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

std::vector<PartitionId> KdTree::roots() const {
    if (nodes_.empty()) return {};
    return {0};
}

std::vector<PartitionId> KdTree::children(PartitionId id) const {
    const Node& node = nodes_.at(id);
    if (node.leaf) return {};
    std::vector<PartitionId> out;
    if (node.left)  out.push_back(*node.left);
    if (node.right) out.push_back(*node.right);
    return out;
}

bool KdTree::is_leaf(PartitionId id) const { return nodes_.at(id).leaf; }

Containment KdTree::classify(PartitionId id, const HyperRect& q) const {
    const Node& node = nodes_.at(id);
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

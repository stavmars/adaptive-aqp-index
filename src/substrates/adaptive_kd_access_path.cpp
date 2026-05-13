#include "a3i/substrates/adaptive_kd_access_path.hpp"

#include <stdexcept>
#include <utility>

namespace a3i {

AdaptiveKdAccessPath::AdaptiveKdAccessPath(SubstrateConfig config)
    : config_(std::move(config)) {}

void AdaptiveKdAccessPath::prepare(IndexTable& table) {
    table_ = &table;
    built_ = false;
    nodes_.clear();
}

void AdaptiveKdAccessPath::ensure_built() {
    if (built_) return;
    if (table_ == nullptr) {
        throw std::logic_error("AdaptiveKdAccessPath::ensure_built before prepare");
    }
    // One root partition owning the whole table; its bounds are the domain.
    Node root;
    root.id     = 0;
    root.bounds = config_.domain_bounds;
    root.begin  = 0;
    root.end    = static_cast<IndexPos>(table_->size());
    root.leaf   = true;
    root.active = true;
    nodes_.clear();
    nodes_.push_back(std::move(root));
    built_ = true;
}

void AdaptiveKdAccessPath::descend(PartitionId id,
                                   const HyperRect& q,
                                   QueryPartitionSet& out) const {
    const Node& node = nodes_[id];
    if (!node.bounds.intersects(q)) return;  // disjoint: in neither list
    if (!node.leaf) {
        if (node.left)  descend(*node.left, q, out);
        if (node.right) descend(*node.right, q, out);
        return;
    }
    if (q.contains_rect(node.bounds)) {
        out.fully_contained.push_back(node.id);
    } else {
        out.partial.push_back(node.id);
    }
}

QueryPartitionSet AdaptiveKdAccessPath::locate(const HyperRect& q) const {
    QueryPartitionSet out;
    if (!nodes_.empty()) descend(/*root=*/0, q, out);
    return out;
}

IndexPos AdaptiveKdAccessPath::hoare_partition(IndexPos start, IndexPos end,
                                               DimensionId axis, double pivot) {
    // Signed cursors so the right cursor can fall below `start` without
    // unsigned wraparound when every point belongs to the >= group.
    std::int64_t i = static_cast<std::int64_t>(start);
    std::int64_t j = static_cast<std::int64_t>(end) - 1;
    while (i <= j) {
        if (table_->dim(static_cast<IndexPos>(i), axis) < pivot) {
            ++i;
        } else {
            while (j >= i && table_->dim(static_cast<IndexPos>(j), axis) >= pivot) {
                --j;
            }
            if (i < j) {
                table_->swap_positions(static_cast<IndexPos>(i),
                                       static_cast<IndexPos>(j));
                ++i;
                --j;
            }
        }
    }
    return static_cast<IndexPos>(i);
}

std::optional<std::pair<PartitionId, PartitionId>>
AdaptiveKdAccessPath::crack_node(PartitionId id, DimensionId axis, double pivot) {
    const IndexPos begin = nodes_[id].begin;
    const IndexPos end   = nodes_[id].end;
    const IndexPos split = hoare_partition(begin, end, axis, pivot);
    if (split == begin || split == end) {
        return std::nullopt;  // failed crack: every point fell on one side
    }

    const PartitionId left_id  = static_cast<PartitionId>(nodes_.size());
    const PartitionId right_id = left_id + 1;

    // Children inherit the parent's bounds, clipped at the pivot on `axis`
    // (half-open: left owns [low, pivot), right owns [pivot, high)).
    Node left;
    left.id              = left_id;
    left.bounds          = nodes_[id].bounds;
    left.bounds.dims[axis].high = pivot;
    left.begin           = begin;
    left.end             = split;
    left.leaf            = true;
    left.active          = true;
    left.parent          = id;

    Node right;
    right.id             = right_id;
    right.bounds         = nodes_[id].bounds;
    right.bounds.dims[axis].low = pivot;
    right.begin          = split;
    right.end            = end;
    right.leaf           = true;
    right.active         = true;
    right.parent         = id;

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

PartitionId AdaptiveKdAccessPath::crack_partition_to_query(PartitionId id,
                                                           const HyperRect& q,
                                                           std::vector<PartitionId>& retired) {
    const DimensionId d = static_cast<DimensionId>(q.dims.size());
    PartitionId cur = id;

    // Each lower bound trims off the points below the query; keep the >= child.
    for (DimensionId axis = 0; axis < d; ++axis) {
        if (nodes_[cur].end - nodes_[cur].begin <= config_.refinement_threshold) return cur;
        auto split = crack_node(cur, axis, q.dims[axis].low);
        if (split) {
            retired.push_back(cur);
            cur = split->second;
        }
    }
    // Each upper bound trims off the points at or above the query; keep the < child.
    for (DimensionId axis = 0; axis < d; ++axis) {
        if (nodes_[cur].end - nodes_[cur].begin <= config_.refinement_threshold) return cur;
        auto split = crack_node(cur, axis, q.dims[axis].high);
        if (split) {
            retired.push_back(cur);
            cur = split->first;
        }
    }
    return cur;
}

RefineResult AdaptiveKdAccessPath::refine(const HyperRect& q, IndexTable& table) {
    ensure_built();
    if (&table != table_) {
        throw std::invalid_argument("AdaptiveKdAccessPath::refine table differs from prepared table");
    }

    RefineResult result;
    // One structural descent: classify the current active frontier. The
    // post-refinement frontier is then derived from this without a second
    // descent, because cracking only ever transforms a partial partition.
    const QueryPartitionSet located = locate(q);

    // Fully-contained partitions are never cracked, so they carry straight
    // over into the new frontier unchanged.
    result.frontier.fully_contained = located.fully_contained;

    for (PartitionId pid : located.partial) {
        if (nodes_[pid].end - nodes_[pid].begin <= config_.refinement_threshold) {
            // Too small to crack: stays a boundary partition on the frontier.
            result.frontier.partial.push_back(pid);
            continue;
        }
        // Crack toward the query. Every slab the crack discards lies wholly
        // outside `q` on some axis (disjoint), so the only frontier member
        // this partition contributes is its single surviving boundary child.
        const PartitionId survivor =
            crack_partition_to_query(pid, q, result.retired);
        if (q.contains_rect(nodes_[survivor].bounds)) {
            result.frontier.fully_contained.push_back(survivor);
        } else {
            result.frontier.partial.push_back(survivor);
        }
    }
    return result;
}

PartitionView AdaptiveKdAccessPath::partition(PartitionId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("AdaptiveKdAccessPath::partition unknown id");
    }
    const Node& node = nodes_[id];
    return PartitionView{node.id, node.bounds, node.begin, node.end, node.active};
}

std::vector<PartitionId> AdaptiveKdAccessPath::active_partitions() const {
    std::vector<PartitionId> ids;
    for (const Node& node : nodes_) {
        if (node.active && node.leaf) ids.push_back(node.id);
    }
    return ids;
}

std::optional<PartitionId> AdaptiveKdAccessPath::parent(PartitionId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("AdaptiveKdAccessPath::parent unknown id");
    }
    return nodes_[id].parent;
}

}  // namespace a3i

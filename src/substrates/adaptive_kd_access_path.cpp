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

std::vector<PartitionId> AdaptiveKdAccessPath::refine(const HyperRect&, IndexTable&) {
    // Query-bound cracking is not yet implemented; the structure does not
    // refine and no parent retires.
    return {};
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

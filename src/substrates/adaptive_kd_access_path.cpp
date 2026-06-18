#include "a3i/substrates/adaptive_kd_access_path.hpp"

#include <stdexcept>
#include <utility>

#include "a3i/substrates/kd_crack.hpp"

namespace a3i {

AdaptiveKdAccessPath::AdaptiveKdAccessPath(SubstrateConfig config)
    : config_(std::move(config)) {}

void AdaptiveKdAccessPath::prepare(IndexTable& table) {
    table_ = &table;
    built_ = false;
}

void AdaptiveKdAccessPath::ensure_built() {
    if (built_) return;
    if (table_ == nullptr) {
        throw std::logic_error("AdaptiveKdAccessPath::ensure_built before prepare");
    }
    // One root partition owning the whole table; its bounds span the data,
    // with the exclusive top lifted just above it to keep the tiling half-open.
    tree_.reset(KdTree::compute_root_bounds(config_.data_bounds, *table_),
                static_cast<IndexPos>(table_->size()));
    built_ = true;
}

std::vector<PartitionId> AdaptiveKdAccessPath::roots() const {
    return tree_.roots();
}

std::vector<PartitionId> AdaptiveKdAccessPath::children(PartitionId id) const {
    return tree_.children(id);
}

bool AdaptiveKdAccessPath::is_leaf(PartitionId id) const {
    return tree_.is_leaf(id);
}

Containment AdaptiveKdAccessPath::classify(PartitionId id,
                                           const HyperRect& q) const {
    return tree_.classify(id, q);
}

std::vector<PartitionId> AdaptiveKdAccessPath::refine(PartitionId id,
                                                     const HyperRect& q,
                                                     IndexTable& table) {
    ensure_built();
    if (&table != table_) {
        throw std::invalid_argument("AdaptiveKdAccessPath::refine table differs from prepared table");
    }

    std::vector<PartitionId> retired;
    if (tree_.population(id) <= config_.partition_size) {
        // Too small to crack: the caller treats it as a boundary leaf.
        return retired;
    }
    // Crack toward the query, isolating its rectangle. Every slab the crack
    // discards lies wholly outside `q` on some axis, so the descent that
    // follows reclassifies the resulting children: the discarded slabs as
    // disjoint, the surviving boundary child as contained or a smaller partial.
    crack_to_query(tree_, *table_, id, q, config_.partition_size, retired);
    return retired;
}

PartitionView AdaptiveKdAccessPath::partition(PartitionId id) const {
    return tree_.partition(id);
}

std::vector<PartitionId> AdaptiveKdAccessPath::active_partitions() const {
    return tree_.active_partitions();
}

std::optional<PartitionId> AdaptiveKdAccessPath::parent(PartitionId id) const {
    return tree_.parent(id);
}

}  // namespace a3i

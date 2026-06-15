#include "a3i/substrates/static_kd_access_path.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace a3i {

StaticKdAccessPath::StaticKdAccessPath(SubstrateConfig config)
    : config_(std::move(config)) {}

void StaticKdAccessPath::prepare(IndexTable& table) {
    table_ = &table;
    built_ = false;
}

double StaticKdAccessPath::median_value(PartitionId id, DimensionId axis) const {
    const KdTree::Node& node = tree_.node(id);
    std::vector<double> coords;
    coords.reserve(node.end - node.begin);
    for (IndexPos p = node.begin; p < node.end; ++p) {
        coords.push_back(table_->dim(p, axis));
    }
    auto mid = coords.begin() + coords.size() / 2;
    std::nth_element(coords.begin(), mid, coords.end());
    return *mid;
}

void StaticKdAccessPath::build(PartitionId id, DimensionId axis) {
    if (tree_.population(id) <= static_cast<IndexPos>(config_.partition_size)) {
        return;
    }
    const double pivot = median_value(id, axis);
    auto split = tree_.split_node(*table_, id, axis, pivot);
    if (!split) {
        // Degenerate range (every coordinate on this axis is >= the median),
        // so a value split cannot separate the points: keep it a leaf.
        return;
    }
    const DimensionId next =
        static_cast<DimensionId>((axis + 1) % table_->dimensions());
    build(split->first, next);
    build(split->second, next);
}

void StaticKdAccessPath::ensure_built() {
    if (built_) return;
    if (table_ == nullptr) {
        throw std::logic_error("StaticKdAccessPath::ensure_built before prepare");
    }
    tree_.reset(KdTree::compute_root_bounds(config_.data_bounds, *table_),
                static_cast<IndexPos>(table_->size()));
    if (table_->size() > 0 && table_->dimensions() > 0) {
        build(/*root=*/0, /*axis=*/0);
    }
    built_ = true;
}

std::vector<PartitionId> StaticKdAccessPath::roots() const {
    return tree_.roots();
}

std::vector<PartitionId> StaticKdAccessPath::children(PartitionId id) const {
    return tree_.children(id);
}

bool StaticKdAccessPath::is_leaf(PartitionId id) const {
    return tree_.is_leaf(id);
}

Containment StaticKdAccessPath::classify(PartitionId id,
                                         const HyperRect& q) const {
    return tree_.classify(id, q);
}

std::vector<PartitionId> StaticKdAccessPath::refine(PartitionId /*id*/,
                                                    const HyperRect& /*q*/,
                                                    IndexTable& table) {
    ensure_built();
    if (&table != table_) {
        throw std::invalid_argument("StaticKdAccessPath::refine table differs from prepared table");
    }
    // The structure is fixed: nothing is cracked and no parent retires.
    return {};
}

PartitionView StaticKdAccessPath::partition(PartitionId id) const {
    return tree_.partition(id);
}

std::vector<PartitionId> StaticKdAccessPath::active_partitions() const {
    return tree_.active_partitions();
}

std::optional<PartitionId> StaticKdAccessPath::parent(PartitionId id) const {
    return tree_.parent(id);
}

}  // namespace a3i

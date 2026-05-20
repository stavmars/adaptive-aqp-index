#include "a3i/substrates/adaptive_kd_access_path.hpp"

#include <stdexcept>
#include <utility>

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
    // One root partition owning the whole table; its bounds are the domain.
    tree_.reset(config_.domain_bounds, static_cast<IndexPos>(table_->size()));
    built_ = true;
}

QueryPartitionSet AdaptiveKdAccessPath::locate(const HyperRect& q) const {
    return tree_.locate(q);
}

PartitionId AdaptiveKdAccessPath::crack_partition_to_query(PartitionId id,
                                                           const HyperRect& q,
                                                           std::vector<PartitionId>& retired) {
    const DimensionId d = static_cast<DimensionId>(q.dims.size());
    PartitionId cur = id;

    // Each lower bound trims off the points below the query; keep the >= child.
    for (DimensionId axis = 0; axis < d; ++axis) {
        if (tree_.population(cur) <= config_.refinement_threshold) return cur;
        auto split = tree_.split_node(*table_, cur, axis, q.dims[axis].low);
        if (split) {
            retired.push_back(cur);
            cur = split->second;
        }
    }
    // Each upper bound trims off the points at or above the query; keep the < child.
    for (DimensionId axis = 0; axis < d; ++axis) {
        if (tree_.population(cur) <= config_.refinement_threshold) return cur;
        auto split = tree_.split_node(*table_, cur, axis, q.dims[axis].high);
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
        if (tree_.population(pid) <= config_.refinement_threshold) {
            // Too small to crack: stays a boundary partition on the frontier.
            result.frontier.partial.push_back(pid);
            continue;
        }
        // Crack toward the query. Every slab the crack discards lies wholly
        // outside `q` on some axis (disjoint), so the only frontier member
        // this partition contributes is its single surviving boundary child.
        const PartitionId survivor =
            crack_partition_to_query(pid, q, result.retired);
        if (q.contains_rect(tree_.node(survivor).bounds)) {
            result.frontier.fully_contained.push_back(survivor);
        } else {
            result.frontier.partial.push_back(survivor);
        }
    }
    return result;
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

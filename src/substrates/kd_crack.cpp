#include "a3i/substrates/kd_crack.hpp"

namespace a3i {

PartitionId crack_to_query(KdTree& tree, IndexTable& table, PartitionId id,
                           const HyperRect& q, std::uint32_t max_population,
                           std::vector<PartitionId>& retired) {
    const DimensionId d = static_cast<DimensionId>(q.dims.size());
    PartitionId cur = id;

    // Each lower bound trims off the points below the query; keep the >= child.
    for (DimensionId axis = 0; axis < d; ++axis) {
        if (tree.population(cur) <= max_population) return cur;
        auto split = tree.split_node(table, cur, axis, q.dims[axis].low);
        if (split) {
            retired.push_back(cur);
            cur = split->second;
        }
    }
    // Each upper bound trims off the points at or above the query; keep the < child.
    for (DimensionId axis = 0; axis < d; ++axis) {
        if (tree.population(cur) <= max_population) return cur;
        auto split = tree.split_node(table, cur, axis, q.dims[axis].high);
        if (split) {
            retired.push_back(cur);
            cur = split->first;
        }
    }
    return cur;
}

}  // namespace a3i

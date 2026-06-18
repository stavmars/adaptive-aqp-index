#include "a3i/substrates/grid_akd_access_path.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include "a3i/substrates/kd_crack.hpp"

namespace a3i {

GridAkdAccessPath::GridAkdAccessPath(SubstrateConfig config)
    : config_(std::move(config)) {}

void GridAkdAccessPath::prepare(IndexTable& table) {
    table_ = &table;
    built_ = false;
}

std::uint32_t GridAkdAccessPath::cell_index(DimensionId axis, double v) const {
    if (width_[axis] <= 0.0) return 0;
    const double f = (v - lo_[axis]) / width_[axis];
    if (f <= 0.0) return 0;
    if (f >= static_cast<double>(k_)) return k_ - 1;
    return static_cast<std::uint32_t>(f);  // floor for a non-negative value
}

std::uint32_t GridAkdAccessPath::flatten(const std::vector<std::uint32_t>& idx) const {
    std::uint32_t c = 0;
    for (DimensionId a = 0; a < dims_; ++a) c = c * k_ + idx[a];
    return c;
}

HyperRect GridAkdAccessPath::tile_bounds(std::uint32_t flat_cell) const {
    std::vector<std::uint32_t> idx(dims_);
    for (int a = static_cast<int>(dims_) - 1; a >= 0; --a) {
        idx[a] = flat_cell % k_;
        flat_cell /= k_;
    }
    HyperRect r;
    r.dims.reserve(dims_);
    for (DimensionId a = 0; a < dims_; ++a) {
        const double low = lo_[a] + idx[a] * width_[a];
        // The top tile keeps the domain's exclusive upper edge; interiors tile
        // the axis at exact multiples of the width.
        const double high = (idx[a] + 1 == k_)
                                ? domain_.dims[a].high
                                : lo_[a] + (idx[a] + 1) * width_[a];
        r.dims.push_back(Range{low, high});
    }
    return r;
}

void GridAkdAccessPath::ensure_built() {
    if (built_) return;
    if (table_ == nullptr) {
        throw std::logic_error("GridAkdAccessPath::ensure_built before prepare");
    }

    dims_ = table_->dimensions();
    k_ = std::max<std::uint32_t>(1, config_.partitions_per_dimension);

    // G = k^d, guarded against exceeding the partition-id width.
    std::uint64_t g = 1;
    for (DimensionId a = 0; a < dims_; ++a) {
        g *= k_;
        if (g > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("GridAkdAccessPath: tile count exceeds id range");
        }
    }
    tiles_ = static_cast<std::uint32_t>(g);

    domain_ = KdTree::compute_root_bounds(config_.data_bounds, *table_);
    lo_.assign(dims_, 0.0);
    width_.assign(dims_, 0.0);
    for (DimensionId a = 0; a < dims_; ++a) {
        lo_[a] = domain_.dims[a].low;
        width_[a] = (domain_.dims[a].high - lo_[a]) / static_cast<double>(k_);
    }

    const std::size_t n = table_->size();

    // Tile each point. Positions are row-major here (pos == row_id), so the
    // resulting `cell` is indexed by row_id.
    std::vector<std::uint32_t> cell(n);
    std::vector<std::uint32_t> idx(dims_);
    for (std::size_t pos = 0; pos < n; ++pos) {
        for (DimensionId a = 0; a < dims_; ++a) {
            idx[a] = cell_index(a, table_->dim(static_cast<IndexPos>(pos), a));
        }
        cell[pos] = flatten(idx);
    }

    // Group the points by tile (out of place); offsets[c] is tile c's start.
    const std::vector<IndexPos> offsets = table_->reorder_by_key(cell, tiles_);

    // Install the root (id 0) with one child per tile (ids 1..G).
    std::vector<HyperRect> bounds(tiles_);
    for (std::uint32_t c = 0; c < tiles_; ++c) bounds[c] = tile_bounds(c);
    tree_.reset_with_children(domain_, static_cast<IndexPos>(n), bounds, offsets);

    // RowId -> owning tile id (tile c is node 1 + c).
    owner_.resize(n);
    for (std::size_t r = 0; r < n; ++r) {
        owner_[r] = static_cast<PartitionId>(1 + cell[r]);
    }

    built_ = true;
}

std::vector<PartitionId> GridAkdAccessPath::roots() const { return tree_.roots(); }

std::vector<PartitionId> GridAkdAccessPath::children(PartitionId id) const {
    if (id == 0) {  // root: tiles are addressed positionally, not via the tree
        std::vector<PartitionId> out(tiles_);
        for (std::uint32_t c = 0; c < tiles_; ++c) {
            out[c] = static_cast<PartitionId>(1 + c);
        }
        return out;
    }
    return tree_.children(id);
}

std::vector<PartitionId> GridAkdAccessPath::overlapping_children(
    PartitionId id, const HyperRect& q) const {
    if (id != 0) return tree_.children(id);

    // Root: the tiles whose cell box overlaps q, enumerated arithmetically.
    std::vector<std::uint32_t> lo_idx(dims_), hi_idx(dims_);
    for (DimensionId a = 0; a < dims_; ++a) {
        lo_idx[a] = cell_index(a, q.dims[a].low);
        hi_idx[a] = cell_index(a, q.dims[a].high);
        if (hi_idx[a] < lo_idx[a]) std::swap(lo_idx[a], hi_idx[a]);
    }
    std::vector<PartitionId> out;
    std::vector<std::uint32_t> idx = lo_idx;
    while (true) {
        out.push_back(static_cast<PartitionId>(1 + flatten(idx)));
        int a = static_cast<int>(dims_) - 1;
        for (; a >= 0; --a) {
            if (idx[a] < hi_idx[a]) {
                ++idx[a];
                break;
            }
            idx[a] = lo_idx[a];
        }
        if (a < 0) break;
    }
    return out;
}

bool GridAkdAccessPath::is_leaf(PartitionId id) const { return tree_.is_leaf(id); }

Containment GridAkdAccessPath::classify(PartitionId id, const HyperRect& q) const {
    return tree_.classify(id, q);
}

std::vector<PartitionId> GridAkdAccessPath::refine(PartitionId id,
                                                   const HyperRect& q,
                                                   IndexTable& table) {
    ensure_built();
    if (&table != table_) {
        throw std::invalid_argument("GridAkdAccessPath::refine table differs from prepared table");
    }
    std::vector<PartitionId> retired;
    if (id == 0) return retired;  // the root is not cracked
    if (tree_.population(id) <= config_.partition_size) return retired;
    crack_to_query(tree_, *table_, id, q, config_.partition_size, retired);
    return retired;
}

PartitionView GridAkdAccessPath::partition(PartitionId id) const {
    return tree_.partition(id);
}

std::vector<PartitionId> GridAkdAccessPath::active_partitions() const {
    return tree_.active_partitions();
}

std::optional<PartitionId> GridAkdAccessPath::parent(PartitionId id) const {
    return tree_.parent(id);
}

}  // namespace a3i

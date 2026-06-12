#include "a3i/storage/index_table.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace a3i {

IndexTable IndexTable::from_columns(
    std::span<const std::span<const double>> columns) {
    if (columns.empty()) {
        throw std::invalid_argument("IndexTable::from_columns: no dimension columns provided");
    }
    if (columns.size() > std::numeric_limits<DimensionId>::max()) {
        throw std::invalid_argument("IndexTable::from_columns: dimension count exceeds DimensionId range");
    }

    const std::size_t n = columns.front().size();
    for (const auto& col : columns) {
        if (col.size() != n) {
            throw std::invalid_argument("IndexTable::from_columns: dimension columns have mismatched lengths");
        }
    }
    if (n > std::numeric_limits<RowId>::max()) {
        throw std::invalid_argument("IndexTable::from_columns: row count exceeds RowId range");
    }

    const auto d = static_cast<DimensionId>(columns.size());

    std::vector<double> points(n * d);
    for (std::size_t pos = 0; pos < n; ++pos) {
        for (DimensionId axis = 0; axis < d; ++axis) {
            points[pos * d + axis] = columns[axis][pos];
        }
    }

    std::vector<RowId> row_ids(n);
    std::iota(row_ids.begin(), row_ids.end(), RowId{0});

    return IndexTable(std::move(points), std::move(row_ids), d);
}

IndexTable IndexTable::from_columns(
    const std::vector<std::vector<double>>& columns) {
    // Adapt owned columns to the span core; the spans view the inputs, so no
    // dimension data is duplicated beyond the AoS buffer the core builds.
    std::vector<std::span<const double>> spans;
    spans.reserve(columns.size());
    for (const auto& col : columns) spans.emplace_back(col);
    return from_columns(std::span<const std::span<const double>>(spans));
}

IndexTable::IndexTable(std::vector<double> points,
                       std::vector<RowId> row_ids,
                       DimensionId dimensions)
    : points_(std::move(points)),
      row_ids_(std::move(row_ids)),
      dimensions_(dimensions) {
    if (dimensions_ == 0) {
        throw std::invalid_argument("IndexTable: dimensions must be > 0");
    }
    if (points_.size() % dimensions_ != 0) {
        throw std::invalid_argument("IndexTable: points size not divisible by dimensions");
    }
    if (points_.size() / dimensions_ != row_ids_.size()) {
        throw std::invalid_argument("IndexTable: row_ids count disagrees with points/dimensions");
    }
}

void IndexTable::swap_positions(IndexPos a, IndexPos b) noexcept {
    if (a == b) return;
    const std::size_t base_a = static_cast<std::size_t>(a) * dimensions_;
    const std::size_t base_b = static_cast<std::size_t>(b) * dimensions_;
    for (DimensionId axis = 0; axis < dimensions_; ++axis) {
        std::swap(points_[base_a + axis], points_[base_b + axis]);
    }
    std::swap(row_ids_[a], row_ids_[b]);
}

}  // namespace a3i

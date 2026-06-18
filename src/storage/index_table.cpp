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

IndexTable IndexTable::from_dimension_reader(
    DimensionId dimensions, std::size_t n,
    const std::function<void(DimensionId, std::size_t, std::size_t,
                             std::span<double>)>& read_chunk,
    std::size_t chunk_rows) {
    if (dimensions == 0) {
        throw std::invalid_argument("IndexTable::from_dimension_reader: no dimensions");
    }
    if (n > std::numeric_limits<RowId>::max()) {
        throw std::invalid_argument(
            "IndexTable::from_dimension_reader: row count exceeds RowId range");
    }
    if (chunk_rows == 0) {
        throw std::invalid_argument("IndexTable::from_dimension_reader: chunk_rows must be > 0");
    }

    // Stream every column in row blocks and interleave each block straight into
    // the AoS buffer in row order, so the table is the only full-size dimension
    // allocation -- the transient is just one small chunk per axis.
    std::vector<double> points(n * dimensions);
    std::vector<std::vector<double>> chunk(dimensions,
                                           std::vector<double>(std::min(chunk_rows, n)));
    for (std::size_t base = 0; base < n; base += chunk_rows) {
        const std::size_t count = std::min(chunk_rows, n - base);
        for (DimensionId axis = 0; axis < dimensions; ++axis) {
            read_chunk(axis, base, count,
                       std::span<double>(chunk[axis].data(), count));
        }
        for (std::size_t i = 0; i < count; ++i) {
            double* dst = points.data() + (base + i) * dimensions;
            for (DimensionId axis = 0; axis < dimensions; ++axis) {
                dst[axis] = chunk[axis][i];
            }
        }
    }

    std::vector<RowId> row_ids(n);
    std::iota(row_ids.begin(), row_ids.end(), RowId{0});

    return IndexTable(std::move(points), std::move(row_ids), dimensions);
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

std::vector<IndexPos> IndexTable::reorder_by_key(
    std::span<const std::uint32_t> key, std::size_t num_keys) {
    const std::size_t n = row_ids_.size();
    if (key.size() != n) {
        throw std::invalid_argument("IndexTable::reorder_by_key: key size must equal size()");
    }

    // Histogram into the offset slots, then prefix-sum so offsets[g] is the
    // start of group g and offsets[num_keys] == n.
    std::vector<IndexPos> offsets(num_keys + 1, 0);
    for (std::size_t pos = 0; pos < n; ++pos) {
        const std::uint32_t g = key[pos];
        if (g >= num_keys) {
            throw std::out_of_range("IndexTable::reorder_by_key: key out of range");
        }
        ++offsets[g + 1];
    }
    for (std::size_t g = 0; g < num_keys; ++g) {
        offsets[g + 1] += offsets[g];
    }

    // Scatter each point and its row_id into the group slot, advancing a
    // per-group cursor; the forward sweep keeps order stable within a group.
    std::vector<double> new_points(points_.size());
    std::vector<RowId>  new_row_ids(n);
    std::vector<IndexPos> cursor(offsets.begin(), offsets.end() - 1);
    for (std::size_t pos = 0; pos < n; ++pos) {
        const std::uint32_t g = key[pos];
        const IndexPos dst = cursor[g]++;
        const double* src = points_.data() + pos * dimensions_;
        double* out = new_points.data() + static_cast<std::size_t>(dst) * dimensions_;
        for (DimensionId axis = 0; axis < dimensions_; ++axis) {
            out[axis] = src[axis];
        }
        new_row_ids[dst] = row_ids_[pos];
    }

    points_  = std::move(new_points);
    row_ids_ = std::move(new_row_ids);
    return offsets;
}

}  // namespace a3i

// Mutable in-memory point table holding the dimensions of every object
// plus an aligned RowId payload.
//
// Layout is array-of-structs: each point's d dimension values live in
// one contiguous block (length == dimensions()), and a parallel row_ids
// array carries the base-table row ordinal at the same position. Every
// permutation of the table (grid build, cracking, swap) must move the
// point block and its row_id together so measures can later be gathered
// by row_id without ambiguity.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "a3i/core/types.hpp"

namespace a3i {

class IndexTable {
public:
    /// Build an index table by interleaving SoA dimension columns into
    /// one contiguous AoS buffer; row_ids are initialized to [0..N).
    ///
    /// All columns must have the same length and at least one column
    /// must be provided. The span overload reads the columns in place (e.g.
    /// the store's resident dimension columns).
    static IndexTable from_columns(const std::vector<std::vector<double>>& columns);
    static IndexTable from_columns(std::span<const std::span<const double>> columns);

    /// Build the same interleaved table by streaming the dimension columns in
    /// row blocks of `chunk_rows`, so no full column is ever held -- only a
    /// small per-axis chunk. `read_chunk(axis, row_offset, count, out)` fills
    /// `out[0..count)` with `count` values of column `axis` starting at
    /// `row_offset`, for each axis in `[0, dimensions)`. `chunk_rows` must be
    /// positive.
    static IndexTable from_dimension_reader(
        DimensionId dimensions, std::size_t n,
        const std::function<void(DimensionId, std::size_t, std::size_t,
                                 std::span<double>)>& read_chunk,
        std::size_t chunk_rows = std::size_t{1} << 20);

    /// An empty table over `dimensions` axes (no rows). Its buffers are sized
    /// later by `scatter_grouped_from_reader`.
    static IndexTable empty(DimensionId dimensions);

    /// Low-level constructor. `points` is the interleaved AoS buffer
    /// (length must equal `dimensions * row_ids.size()`); `row_ids`
    /// gives the base-table row id for each point.
    IndexTable(std::vector<double> points,
               std::vector<RowId> row_ids,
               DimensionId dimensions);

    /// Number of points stored.
    std::size_t size() const noexcept { return row_ids_.size(); }

    /// Number of dimensions per point.
    DimensionId dimensions() const noexcept { return dimensions_; }

    /// Contiguous view over the d coordinates of the point at `pos`.
    std::span<const double> point(IndexPos pos) const noexcept {
        return {points_.data() + static_cast<std::size_t>(pos) * dimensions_,
                dimensions_};
    }

    /// Single-coordinate accessor.
    double dim(IndexPos pos, DimensionId d) const noexcept {
        return points_[static_cast<std::size_t>(pos) * dimensions_ + d];
    }

    /// Base-table row id at `pos`.
    RowId row_id(IndexPos pos) const noexcept { return row_ids_[pos]; }

    /// Swap two points in place: the whole d-dimensional block and the
    /// row_id move together (keeps dimensions and row_ids aligned).
    void swap_positions(IndexPos a, IndexPos b) noexcept;

    /// Reorder points and row_ids out of place so positions sharing a key value
    /// become contiguous, groups appear in ascending key order, and order within
    /// a group is preserved. `key[pos]` is the group of the current position
    /// `pos` (size must equal size(); values in [0, num_keys)). Returns the
    /// prefix-sum offsets (size num_keys+1): group g owns [offsets[g],
    /// offsets[g+1]). Allocates one transient copy of the point and row_id
    /// buffers, which is released on return.
    std::vector<IndexPos> reorder_by_key(std::span<const std::uint32_t> key,
                                         std::size_t num_keys);

    /// Populate this table in grouped order directly from a dimension reader,
    /// without ever holding a second full-size point buffer. `cell[pos]` is the
    /// group of base-table row `pos` (size N); `offsets` is its prefix sum
    /// (size groups+1, `offsets[g]` starts group g, `offsets.back() == N`). The
    /// dimension columns are streamed once via `read_chunk` (same contract as
    /// `from_dimension_reader`) and each row is scattered to the next free slot
    /// of its group, yielding the same layout as `reorder_by_key` -- groups
    /// contiguous in ascending order, stable within a group. `points_`/`row_ids_`
    /// are sized to N here; any existing contents are replaced.
    void scatter_grouped_from_reader(
        std::span<const std::uint32_t> cell,
        std::span<const IndexPos> offsets,
        const std::function<void(DimensionId, std::size_t, std::size_t,
                                 std::span<double>)>& read_chunk,
        std::size_t chunk_rows = std::size_t{1} << 20);

    /// Mutable spans for in-place permutation by the grid build and the
    /// cracking algorithm. Sized `size()*dimensions()` and `size()`.
    std::span<double> points() noexcept { return points_; }
    std::span<RowId>  row_ids() noexcept { return row_ids_; }

    /// Const counterparts.
    std::span<const double> points() const noexcept { return points_; }
    std::span<const RowId>  row_ids() const noexcept { return row_ids_; }

private:
    std::vector<double> points_;   // points_[pos*d + axis]
    std::vector<RowId>  row_ids_;  // row_ids_[pos]
    DimensionId dimensions_;
};

}  // namespace a3i

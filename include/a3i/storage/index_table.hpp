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

    /// True iff a flag column is installed (only when a non-zero budget asked
    /// for one). When absent every flag query is a cheap no-op.
    bool flags_enabled() const noexcept { return !flag_words_.empty(); }

    /// True iff the row currently at `pos` is flagged. False when no flag
    /// column is installed.
    bool is_flagged(IndexPos pos) const noexcept {
        if (flag_words_.empty()) return false;
        return (flag_words_[static_cast<std::size_t>(pos) / 64] >>
                (static_cast<std::size_t>(pos) % 64)) &
               std::uint64_t{1};
    }

    /// Invoke `fn(IndexPos)` for every flagged position in [lo, hi) in
    /// ascending order. Word-scans the range and skips empty words, so the cost
    /// is proportional to the range in words plus the flagged positions found.
    template <typename F>
    void for_each_flagged_in_range(IndexPos lo, IndexPos hi, F&& fn) const {
        if (flag_words_.empty() || hi <= lo) return;
        const std::size_t w0 = static_cast<std::size_t>(lo) / 64;
        const std::size_t w1 = static_cast<std::size_t>(hi - 1) / 64;
        for (std::size_t w = w0; w <= w1; ++w) {
            std::uint64_t bits = flag_words_[w];
            if (w == w0) {
                bits &= ~std::uint64_t{0} << (static_cast<std::size_t>(lo) % 64);
            }
            if (w == w1) {
                const unsigned top =
                    static_cast<unsigned>(static_cast<std::size_t>(hi - 1) % 64);
                bits &= (top == 63) ? ~std::uint64_t{0}
                                    : ((std::uint64_t{1} << (top + 1)) - 1);
            }
            while (bits) {
                const unsigned b = static_cast<unsigned>(__builtin_ctzll(bits));
                fn(static_cast<IndexPos>(w * 64 + b));
                bits &= bits - 1;
            }
        }
    }

    /// Install the flag column: mark exactly the positions whose current row_id
    /// appears in `flagged_rowids`. Sizes the column to size() and replaces any
    /// prior column. Call once after the table is in its final order; the flag
    /// then rides every later swap_positions.
    void set_flags_by_rowid(std::span<const RowId> flagged_rowids);

    /// Swap two points in place: the whole d-dimensional block, the row_id, and
    /// the flag bit (when installed) move together so dimensions, row_ids, and
    /// flags stay aligned.
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
    std::vector<std::uint64_t> flag_words_;  // 1 bit/pos; empty when not installed
    DimensionId dimensions_;
};

}  // namespace a3i

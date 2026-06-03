// On-disk binary column store backing a prepared dataset.
//
// One file per dimension or measure column; raw little-endian IEEE-754
// float64, no in-file header — length and metadata come from manifest.json.
// Dimensions load eagerly (the access path needs them in RAM for
// cracking). Measure columns are backed one of two ways, chosen at open time
// (`MeasureStorage`), with an identical by-RowId access contract either way:
//   - Mmap (default, out-of-core): each column is memory-mapped so a lookup is
//     an O(1) page touch (`MADV_RANDOM` by default; callers that scan ranges
//     can promote a region to `MADV_WILLNEED`/`MADV_SEQUENTIAL` via the
//     prefetch helper). This is the only backing that lets a column exceed RAM.
//   - Eager (in-memory): each column is read fully resident at open time, like
//     the dimensions, so a lookup is a plain array index with no page-fault
//     path. Meaningful only when the dataset fits in RAM.
//
// `gather()` reads a batch of measure values in the order the caller asks for.
// Any reordering for read locality (e.g. sorting RowIds ascending, or
// k-way-merging streams from multiple partitions) is the caller's choice and
// happens upstream.

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "a3i/core/types.hpp"
#include "a3i/storage/manifest.hpp"

namespace a3i {

/// How the exposed measure columns are backed. The choice does not affect the
/// values returned by any accessor — only whether reads fault in mmap'd pages
/// or index a resident array (see the file header).
enum class MeasureStorage {
    Mmap,   ///< Memory-map each column (out-of-core; the default).
    Eager,  ///< Load each column fully resident at open time (in-memory).
};

class BinaryColumnStore {
public:
    /// Open a prepared dataset by manifest path. Eagerly loads dimensions
    /// into memory; backs the exposed measure columns per `measure_storage`
    /// (Mmap with MADV_RANDOM by default, or a fully-resident Eager load).
    ///
    /// `selected_measures` chooses which measure columns to expose, in the
    /// given order: the store then reports `measure_count()` of them and the
    /// measure accessors (`measure_value`, `gather`, `global_stats`) take a
    /// local id in `[0, selected_measures.size())`. An empty list (the
    /// default) exposes every measure in manifest order. Each id must be a
    /// valid index into the manifest's measures; the policy that decides the
    /// subset lives entirely in the caller. `manifest()` still returns the
    /// full artifact metadata regardless of the selection.
    ///
    /// `measure_storage` selects the backing for the exposed columns. Eager is
    /// meaningful only when the dataset fits in RAM (it allocates one resident
    /// vector per exposed column); the access contract is otherwise identical.
    explicit BinaryColumnStore(const std::filesystem::path& manifest_path,
                               std::vector<MeasureId> selected_measures = {},
                               MeasureStorage measure_storage = MeasureStorage::Mmap);
    ~BinaryColumnStore();

    BinaryColumnStore(const BinaryColumnStore&) = delete;
    BinaryColumnStore& operator=(const BinaryColumnStore&) = delete;

    /// The whole manifest (cheap accessor; held internally).
    const Manifest& manifest() const noexcept { return manifest_; }

    std::size_t row_count() const noexcept { return manifest_.row_count; }
    std::size_t dimension_count() const noexcept { return manifest_.dimensions.size(); }
    std::size_t measure_count() const noexcept { return exposed_measures_.size(); }

    /// In-memory contiguous view of one dimension column.
    std::span<const double> dimension_column(DimensionId d) const;

    /// O(1) random access to one measure column (a mapped page touch, or a
    /// resident array index when the column is loaded eager).
    double measure_value(RowId r, MeasureId m) const;

    /// Read one measure value per RowId in the order given. `out[i]` receives
    /// the value at `row_ids[i]`; `row_ids` and `out` must have equal length.
    /// No internal reordering: callers that want sequential page access
    /// should sort or k-way-merge their RowIds before calling.
    void gather(MeasureId m,
                std::span<const RowId> row_ids,
                std::span<double> out) const;

    /// Whole measure column as a contiguous span. Under mmap, reading an
    /// element faults its page in on first touch; under eager backing it is
    /// already resident. Intended for one-shot front-to-back sweeps that want
    /// to avoid the per-element bounds checks of `measure_value`/`gather`.
    /// Empty if the column has zero rows.
    std::span<const double> measure_column(MeasureId m) const;

    /// Read-ahead hint for a full front-to-back scan of measure `m`'s column
    /// (kernel `MADV_SEQUENTIAL`): pages may be dropped behind the read cursor,
    /// so use it only for a dense one-pass sweep, never for sampling. A no-op
    /// when the column is held resident (nothing is mapped).
    void advise_sequential(MeasureId m) const;

    /// Per-measure stats from the manifest (no scan).
    const GlobalMeasureStats& global_stats(MeasureId m) const;

private:
    // Memory-mapped backing for one measure column (Mmap mode only). Kept so
    // the mapping can be advised and unmapped; the hot-path handle that reads
    // give out lives in measure_views_.
    struct MmapRegion {
        void*       base = nullptr;
        std::size_t bytes = 0;
        int         fd = -1;
    };

    // Backing-agnostic, hot-path handle into one exposed measure column,
    // whether it is memory-mapped or held resident.
    struct MeasureView {
        const double* data = nullptr;
        std::size_t   length = 0;   ///< In elements, not bytes.
    };

    Manifest manifest_;
    std::filesystem::path manifest_dir_;
    std::vector<std::vector<double>> dim_columns_;       ///< Eager copies.
    MeasureStorage                   storage_mode_ = MeasureStorage::Mmap;
    std::vector<MmapRegion>          measure_maps_;      ///< Mmap mode: one per exposed measure.
    std::vector<std::vector<double>> measure_resident_;  ///< Eager mode: one per exposed measure.
    std::vector<MeasureView>         measure_views_;     ///< Always: hot-path handle per exposed measure.
    std::vector<MeasureId>           exposed_measures_;  ///< Local id -> manifest measure index.
};

}  // namespace a3i

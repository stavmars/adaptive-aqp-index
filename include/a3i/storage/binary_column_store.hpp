// On-disk binary column store backing a prepared dataset.
//
// One file per dimension or measure column; raw little-endian IEEE-754
// float64, no in-file header — length and metadata come from manifest.json.
// Dimensions load eagerly (the access path needs them in RAM for
// cracking). Measures are memory-mapped so a lookup is an O(1) page touch
// (`MADV_RANDOM` by default; callers that scan ranges can promote a region
// to `MADV_WILLNEED`/`MADV_SEQUENTIAL` via prefetch helpers added later).
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

class BinaryColumnStore {
public:
    /// Open a prepared dataset by manifest path. Eagerly loads dimensions
    /// into memory; mmaps every measure column with MADV_RANDOM.
    ///
    /// `selected_measures` chooses which measure columns to expose, in the
    /// given order: the store then reports `measure_count()` of them and the
    /// measure accessors (`measure_value`, `gather`, `global_stats`) take a
    /// local id in `[0, selected_measures.size())`. An empty list (the
    /// default) exposes every measure in manifest order. Each id must be a
    /// valid index into the manifest's measures; the policy that decides the
    /// subset lives entirely in the caller. `manifest()` still returns the
    /// full artifact metadata regardless of the selection.
    explicit BinaryColumnStore(const std::filesystem::path& manifest_path,
                               std::vector<MeasureId> selected_measures = {});
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

    /// O(1) random access on a memory-mapped measure column.
    double measure_value(RowId r, MeasureId m) const;

    /// Read one measure value per RowId in the order given. `out[i]` receives
    /// the value at `row_ids[i]`; `row_ids` and `out` must have equal length.
    /// No internal reordering: callers that want sequential page access
    /// should sort or k-way-merge their RowIds before calling.
    void gather(MeasureId m,
                std::span<const RowId> row_ids,
                std::span<double> out) const;

    /// Whole memory-mapped measure column as a contiguous span. Reading an
    /// element faults its page in on first touch. Intended for one-shot
    /// front-to-back sweeps that want to avoid the per-element bounds checks of
    /// `measure_value`/`gather`. Empty if the column is unmapped (zero rows).
    std::span<const double> measure_column(MeasureId m) const;

    /// Read-ahead hint for a full front-to-back scan of measure `m`'s column
    /// (kernel `MADV_SEQUENTIAL`): pages may be dropped behind the read cursor,
    /// so use it only for a dense one-pass sweep, never for sampling.
    void advise_sequential(MeasureId m) const;

    /// Per-measure stats from the manifest (no scan).
    const GlobalMeasureStats& global_stats(MeasureId m) const;

private:
    struct MmapRegion {
        const double* data = nullptr;
        std::size_t   length = 0;   ///< In elements, not bytes.
        void*         base = nullptr;
        std::size_t   bytes = 0;
        int           fd = -1;
    };

    Manifest manifest_;
    std::filesystem::path manifest_dir_;
    std::vector<std::vector<double>> dim_columns_;  ///< Eager copies.
    std::vector<MmapRegion> measure_regions_;       ///< One per exposed measure.
    std::vector<MeasureId>  exposed_measures_;      ///< Local id -> manifest measure index.
};

}  // namespace a3i

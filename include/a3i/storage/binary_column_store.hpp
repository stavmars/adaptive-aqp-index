// On-disk binary column store backing a prepared dataset.
//
// One file per dimension or measure column; raw little-endian IEEE-754
// float64, no in-file header — length and metadata come from manifest.json.
// Dimensions load eagerly (the access path needs them in RAM for
// cracking). Measure columns are backed one of two ways, chosen at open time
// (`MeasureStorage`), with an identical by-RowId access contract either way:
//   - OnDisk (default, out-of-core): each column stays on disk behind an open
//     file descriptor and is read with explicit I/O. Scattered batches
//     (`gather`) are sorted, coalesced into page-aligned ranges, and issued
//     as asynchronous batched reads so many requests are in flight at once —
//     the queue depth SSDs need to deliver their random-read throughput.
//     Sequential block reads (`read_rows`) use plain positional reads that
//     the kernel's readahead streams at full bandwidth. This is the only
//     backing that lets a column exceed RAM.
//   - Eager (in-memory): each column is read fully resident at open time, like
//     the dimensions, so a lookup is a plain array index with no I/O path.
//     Meaningful only when the dataset fits in RAM.
//
// `gather()` reads a batch of measure values and returns them in the order
// the caller asked for (`out[i]` = value at `row_ids[i]`), regardless of how
// the reads are scheduled internally.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "a3i/core/types.hpp"
#include "a3i/storage/manifest.hpp"

namespace a3i {

/// How the exposed measure columns are backed. The choice does not affect the
/// values returned by any accessor — only whether reads go to disk through
/// explicit I/O or index a resident array (see the file header).
enum class MeasureStorage {
    OnDisk,  ///< Keep each column on disk; explicit batched I/O (the default).
    Eager,   ///< Load each column fully resident at open time (in-memory).
};

/// How a single on-disk batch was served, split by access path (see the
/// cost model in the .cpp). A batch picks one path for all its measures.
/// `*_rows` count the distinct WANTED rows served by each path (row, not
/// row*measure). `*_bytes` count the actual device bytes moved by each path
/// across all measures -- much larger than the wanted rows imply, because a
/// scan reads its whole row span and a gather reads whole pages. All stay
/// zero for an eager (in-memory) store. Callers accumulate these across a
/// query to report the scan/gather mix and the real I/O it cost.
struct GatherPathStats {
    std::uint64_t scan_rows    = 0;  ///< Wanted rows served by the scan path.
    std::uint64_t gather_rows  = 0;  ///< Wanted rows served by the gather path.
    std::uint64_t scan_bytes   = 0;  ///< Device bytes read by the scan path.
    std::uint64_t gather_bytes = 0;  ///< Device bytes read by the gather path.
};

class BinaryColumnStore {
public:
    /// Open a prepared dataset by manifest path. Eagerly loads dimensions
    /// into memory; backs the exposed measure columns per `measure_storage`.
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
                               MeasureStorage measure_storage = MeasureStorage::OnDisk);
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

    /// Random access to one measure value (a resident array index when the
    /// column is loaded eager, a single positional read when on disk). For
    /// more than a handful of rows, `gather` is the right call.
    double measure_value(RowId r, MeasureId m) const;

    /// Read one measure value per RowId; `out[i]` receives the value at
    /// `row_ids[i]`. `row_ids` and `out` must have equal length. Ids may be
    /// in any order and may repeat. On-disk reads are internally sorted,
    /// coalesced into page-aligned ranges, and issued as a batch of
    /// asynchronous reads with many requests in flight; results are scattered
    /// back into caller order. Not safe to call concurrently on one store.
    ///
    /// `already_sorted` lets a caller that knows its ids are ascending (e.g.
    /// the engine's k-way merge output when gather-sorting is on) skip the
    /// internal order check; passing it wrongly produces incorrect coalescing,
    /// so leave it false unless the ids are provably sorted.
    void gather(MeasureId m,
                std::span<const RowId> row_ids,
                std::span<double> out,
                bool already_sorted = false) const;

    /// Read every exposed measure at each RowId: `outs[m][i]` receives
    /// measure `m`'s value at `row_ids[i]` (`outs` is resized to
    /// `measure_count()` vectors of `row_ids.size()`). On disk, all measures'
    /// reads share one submission schedule, so the device sees one deep queue
    /// instead of one drain-and-refill pass per measure. Same ordering and
    /// concurrency contract as `gather`. `already_sorted` is as in `gather`.
    ///
    /// If `path_stats` is non-null it receives how this batch was served,
    /// split by access path (scan vs scattered gather); see `GatherPathStats`.
    /// It is left untouched for an eager store (no I/O path). Telemetry only —
    /// the values returned are identical with or without it.
    void gather_all(std::span<const RowId> row_ids,
                    std::vector<std::vector<double>>& outs,
                    bool already_sorted = false,
                    GatherPathStats* path_stats = nullptr) const;

    /// Read `count` consecutive rows of measure `m` starting at `begin`.
    /// Returns a view of the values: a zero-copy subspan of the resident
    /// column under eager backing, or `scratch` filled by sequential
    /// positional reads when on disk (resized as needed). Intended for
    /// front-to-back block sweeps; pair with `advise_sequential`.
    std::span<const double> read_rows(MeasureId m, RowId begin,
                                      std::size_t count,
                                      std::vector<double>& scratch) const;

    /// Whole measure column as a contiguous span. Only valid under eager
    /// backing (the data is resident); throws `std::logic_error` for an
    /// on-disk column — sweep those in blocks with `read_rows` instead.
    std::span<const double> measure_column(MeasureId m) const;

    /// Read-ahead hint for a full front-to-back sweep of measure `m`'s column
    /// (kernel sequential-access advice on the file descriptor). A no-op when
    /// the column is held resident.
    void advise_sequential(MeasureId m) const;

    /// Per-measure stats from the manifest (no scan).
    const GlobalMeasureStats& global_stats(MeasureId m) const;

private:
    // On-disk backing for one measure column: an open read-only descriptor
    // plus the exact payload size (rows * sizeof(double)), which bounds every
    // read so a truncated file is caught rather than silently zero-filled.
    struct OnDiskColumn {
        int           fd = -1;
        std::uint64_t bytes = 0;
    };

    struct IoRing;  // Batched-read submission context (defined in the .cpp).

    Manifest manifest_;
    std::filesystem::path manifest_dir_;
    std::vector<std::vector<double>> dim_columns_;       ///< Eager copies.
    MeasureStorage                   storage_mode_ = MeasureStorage::OnDisk;
    std::vector<OnDiskColumn>        measure_files_;     ///< OnDisk mode: one per exposed measure.
    std::vector<std::vector<double>> measure_resident_;  ///< Eager mode: one per exposed measure.
    std::vector<MeasureId>           exposed_measures_;  ///< Local id -> manifest measure index.

    // Submission context for batched on-disk gathers. Lazily initialized on
    // first use; null if the kernel interface is unavailable, in which case
    // gathers fall back to synchronous positional reads over the same
    // coalesced ranges. Mutable: scheduling reads does not change any value
    // an accessor returns.
    mutable std::unique_ptr<IoRing> ring_;
    mutable bool                    ring_init_attempted_ = false;

    // Shared on-disk batch path: one coalesced read plan over the (sorted)
    // row ids, executed for each requested measure column; `outs[c][i]`
    // receives measure `ms[c]`'s value at `row_ids[i]`. If `path_stats` is
    // non-null, records which access path this batch took (scan vs gather).
    void gather_batch(std::span<const MeasureId> ms,
                      std::span<const RowId> row_ids,
                      double* const* outs,
                      bool already_sorted,
                      GatherPathStats* path_stats = nullptr) const;

    // Sequential-scan path for gather_batch: when the wanted rows are dense
    // enough over their [lo, lo+span_rows) span that a streaming scan beats a
    // scattered gather (see kRandomVsSequentialBw), read the span in blocks and
    // pick the wanted values. `order` is the ascending permutation of `row_ids`
    // (empty when `row_ids` is already ascending); `outs[c][orig]` receives
    // measure `ms[c]`'s value at the original caller position. Returns the
    // total device bytes read (summed over all measure columns).
    std::uint64_t scan_span(std::span<const MeasureId> ms,
                            std::span<const RowId> row_ids,
                            double* const* outs,
                            std::span<const std::size_t> order,
                            RowId lo, std::uint64_t span_rows) const;
};

}  // namespace a3i

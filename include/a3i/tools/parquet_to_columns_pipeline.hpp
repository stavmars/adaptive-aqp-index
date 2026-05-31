// Offline Parquet -> binary column converter pipeline.
//
// Reads the dataset's Parquet (typed, columnar, unfiltered),
// projects only the requested dimensions and measures by name, encodes
// missing/null measure values as NaN, applies the row survival rule
// (domain bounds plus drop filters), and writes one .bin per kept column
// plus a manifest.json describing them.
//
// All decisions about *what* to convert live here. The CLI wrapper in
// tools/convert_parquet_to_columns.cpp only parses arguments and calls
// `run_parquet_to_columns()`. This split lets unit tests drive the pipeline
// in-process against tiny fixtures without exec().
//
// Never called from the runner. Idempotent across runs with the same
// arguments; deterministic (RowId == position in the post-filter sequence).

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace a3i {

/// One requested dimension column.
struct DimensionRequest {
    std::string name;   ///< Resolved against the Parquet schema's column names.
    double low  = 0.0;  ///< Domain bound (low, inclusive).
    double high = 0.0;  ///< Domain bound (high, exclusive — half-open interval).
};

/// One row-level drop filter, e.g. {"DURATION_MINUTES", Gt, 1440.0}.
/// Operates on raw numeric values of any dimension or measure column the
/// converter reads. A row is dropped if any filter matches.
struct ValidationFilter {
    enum class Op { Lt, Le, Gt, Ge, Eq, Ne };
    std::string name;
    Op          op = Op::Gt;
    double      value = 0.0;
};

struct ConvertOptions {
    std::filesystem::path input_parquet;   ///< Required: the Parquet source.
    std::filesystem::path output_dir;      ///< Required: where columns/ + manifest.json land.
    std::string           dataset_id;      ///< Required: copied into the manifest.
    std::vector<DimensionRequest>   dimensions;
    std::vector<std::string>        measures;
    std::vector<ValidationFilter>   validation_filters;
    std::optional<std::uint64_t>    max_rows;       ///< Take only the first N post-filter rows.
    bool                            overwrite = false;
    std::string                     converter_version = "0.1.0";
};

/// Outcome of one conversion. Numbers are post-filter (i.e. what landed
/// in the .bin files).
struct ConvertReport {
    std::uint64_t rows_written = 0;
    std::uint64_t rows_read    = 0;          ///< Total source rows scanned (incl. dropped).
    std::uint64_t rows_filtered_out = 0;     ///< Dropped by filters or out-of-bounds.
    std::filesystem::path manifest_path;
};

/// Run one conversion. Throws std::runtime_error on I/O errors,
/// std::invalid_argument on unresolved names or malformed options.
ConvertReport run_parquet_to_columns(const ConvertOptions& opts);

}  // namespace a3i

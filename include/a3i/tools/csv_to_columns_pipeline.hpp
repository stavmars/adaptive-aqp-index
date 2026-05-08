// Offline CSV -> binary column converter pipeline.
//
// Reads a (header or headerless) delimited text file, projects only the
// requested dimensions and measures, encodes missing/invalid measure
// values as NaN, applies optional row-level validation filters, and
// writes one .bin per kept column plus a manifest.json describing them.
//
// All decisions about *what* to convert live here. The CLI wrapper in
// tools/convert_csv_to_columns.cpp only parses arguments and calls
// `run_csv_to_columns()`. This split lets unit tests drive the pipeline
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
    std::string name;   ///< From the header, or DuckDB-style "columnNN" fallback.
    double low  = 0.0;  ///< Domain bound (low, inclusive).
    double high = 0.0;  ///< Domain bound (high, exclusive — half-open interval).
};

/// One row-level validation filter, e.g. {"DURATION_MINUTES", Gt, 1440.0}.
/// Operates on raw numeric values of any dimension or measure column the
/// converter parses (after `null_string` masking but before dropping NaNs).
struct ValidationFilter {
    enum class Op { Lt, Le, Gt, Ge, Eq, Ne };
    std::string name;
    Op          op = Op::Gt;
    double      value = 0.0;
};

struct ConvertOptions {
    std::filesystem::path input_csv;       ///< Required.
    std::filesystem::path output_dir;      ///< Required: where columns/ + manifest.json land.
    std::string           dataset_id;      ///< Required: copied into the manifest.
    bool                  has_header = false;
    char                  delimiter  = ',';
    std::string           null_string;     ///< Empty => no null sentinel (only empty fields are NaN).
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
    std::uint64_t rows_read    = 0;          ///< Total CSV rows scanned (incl. dropped).
    std::uint64_t rows_filtered_out = 0;     ///< Dropped by validation_filters or parse failures.
    std::filesystem::path manifest_path;
};

/// Synthesize DuckDB-style column names for a headerless CSV with
/// `num_columns` columns: "column" + zero-padded zero-based index,
/// padded to `len(str(num_columns - 1))` digits. So 1..9 columns yield
/// `column0..column8`; 10..99 columns yield `column00..column99`.
std::vector<std::string> duckdb_default_column_names(std::size_t num_columns);

/// Run one conversion. Throws std::runtime_error on I/O errors,
/// std::invalid_argument on unresolved names or malformed options.
ConvertReport run_csv_to_columns(const ConvertOptions& opts);

}  // namespace a3i

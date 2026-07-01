// One-off CSV/TSV -> Parquet conversion.
//
// This is the heavy, run-once text-wrangling step: it reads a delimited
// text file (with or without a header), infers a faithful column type for
// each field, and writes a typed, columnar Parquet artifact. No projection
// and no row filtering happen here -- the output is the immutable,
// unfiltered universe that the binary converter (and external systems)
// later consume.
//
// The Parquet produced here is the single interchange format for the
// experiment pipeline: both A3I's binary converter and the external
// baseline systems read this same file, so every system sees identical,
// already-typed rows and nothing re-parses text.
//
// Column names come from the header row when present; a headerless source
// has none to carry over, so this step synthesizes positional `col0`,
// `col1`, ... names into the schema.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace a3i {

struct CsvToParquetOptions {
    std::filesystem::path input_path;   ///< Required: the source CSV/TSV.
    std::filesystem::path output_path;  ///< Required: the .parquet to write.
    bool        has_header = false;     ///< First row holds column names.
    char        delimiter  = ',';       ///< Field separator (',' or '\t').
    std::string null_string;            ///< Extra null sentinel; empty fields are always null.
    bool        overwrite  = false;     ///< Replace an existing output_path.
    /// Extra strptime patterns to recognize timestamp columns, tried in order
    /// (ISO8601 is always tried as well). A column typed as a timestamp only
    /// when every value matches one of these or ISO8601; this lets a column
    /// whose rows mix several datetime formats still infer as a timestamp
    /// instead of falling back to string. Empty keeps the default ISO8601-only
    /// inference.
    std::vector<std::string> timestamp_formats;
};

struct CsvToParquetReport {
    std::uint64_t            rows = 0;
    std::vector<std::string> column_names;  ///< Header names or synthesized fallbacks.
};

/// Convert a delimited text file to a typed Parquet file. Column
/// types are inferred by the CSV reader (integers, doubles, booleans,
/// timestamps, strings); extra timestamp patterns can be supplied via
/// opts.timestamp_formats. Throws std::runtime_error on I/O or parse errors,
/// std::invalid_argument on malformed options.
CsvToParquetReport csv_to_parquet(const CsvToParquetOptions& opts);

/// Synthesize positional column names for `num_columns` columns: "col"
/// + zero-padded zero-based index, padded to `len(str(num_columns - 1))`
/// digits. So 1..9 columns yield `col0..col8`; 10..99 columns yield
/// `col00..col99`. Used to name the columns of a headerless source,
/// which carries no names of its own.
std::vector<std::string> default_column_names(std::size_t num_columns);

}  // namespace a3i

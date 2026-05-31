#include "a3i/tools/csv_to_parquet.hpp"

#include <stdexcept>
#include <string>

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

namespace a3i {

namespace {

// Translate a failed Arrow Status into an exception the CLI/tests expect.
template <class T>
T unwrap(arrow::Result<T> result, const std::string& what) {
    if (!result.ok()) {
        throw std::runtime_error(what + ": " + result.status().ToString());
    }
    return std::move(result).ValueUnsafe();
}

void check(const arrow::Status& status, const std::string& what) {
    if (!status.ok()) {
        throw std::runtime_error(what + ": " + status.ToString());
    }
}

}  // namespace

std::vector<std::string> default_column_names(std::size_t num_columns) {
    if (num_columns == 0) return {};
    const std::size_t max_idx = num_columns - 1;
    const std::size_t width = std::to_string(max_idx).size();
    std::vector<std::string> names;
    names.reserve(num_columns);
    for (std::size_t i = 0; i < num_columns; ++i) {
        auto s = std::to_string(i);
        if (s.size() < width) s.insert(0, width - s.size(), '0');
        names.push_back("col" + s);
    }
    return names;
}

CsvToParquetReport csv_to_parquet(const CsvToParquetOptions& opts) {
    if (opts.input_path.empty())  throw std::invalid_argument("input_path is required");
    if (opts.output_path.empty()) throw std::invalid_argument("output_path is required");
    if (opts.delimiter != ',' && opts.delimiter != '\t') {
        throw std::invalid_argument(
            std::string("unsupported delimiter: '") + opts.delimiter +
            "' (only ',' and '\\t' are supported)");
    }
    if (!opts.overwrite && std::filesystem::exists(opts.output_path)) {
        throw std::runtime_error("output already exists (pass overwrite to replace): " +
                                 opts.output_path.string());
    }
    if (opts.output_path.has_parent_path()) {
        std::filesystem::create_directories(opts.output_path.parent_path());
    }

    auto input = unwrap(arrow::io::ReadableFile::Open(opts.input_path.string()),
                        "cannot open input csv: " + opts.input_path.string());

    auto read_opts  = arrow::csv::ReadOptions::Defaults();
    // For a headerless file we let the reader treat the first line as data
    // and emit positional names, then rename to stable `colNN` names below.
    read_opts.autogenerate_column_names = !opts.has_header;

    auto parse_opts     = arrow::csv::ParseOptions::Defaults();
    parse_opts.delimiter = opts.delimiter;

    auto convert_opts = arrow::csv::ConvertOptions::Defaults();
    convert_opts.strings_can_be_null = true;
    if (!opts.null_string.empty()) {
        convert_opts.null_values.push_back(opts.null_string);
    }

    auto reader = unwrap(
        arrow::csv::TableReader::Make(arrow::io::default_io_context(), input,
                                      read_opts, parse_opts, convert_opts),
        "cannot build csv reader");
    auto table = unwrap(reader->Read(), "cannot read csv: " + opts.input_path.string());

    if (!opts.has_header) {
        auto names = default_column_names(
            static_cast<std::size_t>(table->num_columns()));
        table = unwrap(table->RenameColumns(names), "cannot rename columns");
    }

    auto out = unwrap(arrow::io::FileOutputStream::Open(opts.output_path.string()),
                      "cannot open output parquet: " + opts.output_path.string());

    // Pin writer properties so the same input yields a byte-identical file:
    // single row group, no compression, stable encodings.
    parquet::WriterProperties::Builder builder;
    builder.compression(parquet::Compression::UNCOMPRESSED);
    builder.disable_dictionary();
    builder.version(parquet::ParquetVersion::PARQUET_2_6);
    auto props = builder.build();

    const std::int64_t chunk = table->num_rows() > 0 ? table->num_rows() : 1;
    check(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                     out, chunk, props),
          "cannot write parquet: " + opts.output_path.string());
    check(out->Close(), "cannot close parquet output");

    CsvToParquetReport rep;
    rep.rows = static_cast<std::uint64_t>(table->num_rows());
    rep.column_names.reserve(static_cast<std::size_t>(table->num_columns()));
    for (const auto& field : table->schema()->fields()) {
        rep.column_names.push_back(field->name());
    }
    return rep;
}

}  // namespace a3i

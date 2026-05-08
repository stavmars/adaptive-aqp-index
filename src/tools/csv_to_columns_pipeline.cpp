#include "a3i/tools/csv_to_columns_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <csv2/reader.hpp>
#include <fast_float/fast_float.h>

#include "a3i/core/geometry.hpp"
#include "a3i/storage/manifest.hpp"

namespace a3i {

namespace {

// --- Numeric parsing ----------------------------------------------------

bool parse_double(std::string_view sv, double& out) {
    if (sv.empty()) return false;
    const auto res = fast_float::from_chars(sv.data(), sv.data() + sv.size(), out);
    return res.ec == std::errc{} && res.ptr == sv.data() + sv.size();
}

// --- Filter semantics ---------------------------------------------------
// A filter like "DURATION_MINUTES > 1440" *drops* matching rows; NaN
// observations never cause a row to be dropped on numeric grounds.
bool keep_under(const ValidationFilter::Op op, double cell, double rhs) {
    if (std::isnan(cell)) return true;
    switch (op) {
        case ValidationFilter::Op::Lt: return !(cell <  rhs);
        case ValidationFilter::Op::Le: return !(cell <= rhs);
        case ValidationFilter::Op::Gt: return !(cell >  rhs);
        case ValidationFilter::Op::Ge: return !(cell >= rhs);
        case ValidationFilter::Op::Eq: return !(cell == rhs);
        case ValidationFilter::Op::Ne: return !(cell != rhs);
    }
    return true;
}

std::string utc_iso8601_now() {
    using namespace std::chrono;
    const auto now = system_clock::to_time_t(system_clock::now());
    std::tm tm{};
    gmtime_r(&now, &tm);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02dZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

struct ColumnSink {
    std::ofstream out;
    std::filesystem::path path;

    explicit ColumnSink(const std::filesystem::path& p)
        : out(p, std::ios::binary | std::ios::trunc), path(p) {
        if (!out) throw std::runtime_error("cannot open column file: " + p.string());
    }
    void write(double v) {
        out.write(reinterpret_cast<const char*>(&v), sizeof v);
        if (!out) throw std::runtime_error("write failed: " + path.string());
    }
};

struct DimRunning {
    double minv =  std::numeric_limits<double>::infinity();
    double maxv = -std::numeric_limits<double>::infinity();
};

struct MeasRunning {
    std::uint64_t non_nan = 0;
    std::uint64_t nan_n   = 0;
    double sum    = 0.0;
    double sum_sq = 0.0;
    double minv   =  std::numeric_limits<double>::infinity();
    double maxv   = -std::numeric_limits<double>::infinity();
};

// Templated on the csv2 delimiter policy so we can dispatch on a runtime
// delimiter at the entry point. csv2's delimiter is a compile-time
// template parameter; we instantiate one variant per supported delim.
template <class DelimPolicy>
ConvertReport run_with_delim(const ConvertOptions& opts) {
    namespace c2 = csv2;
    using Reader = c2::Reader<DelimPolicy,
                              c2::quote_character<'"'>,
                              c2::first_row_is_header<false>,
                              c2::trim_policy::no_trimming>;

    const auto columns_dir   = opts.output_dir / "columns";
    const auto manifest_path = opts.output_dir / "manifest.json";
    if (!opts.overwrite && std::filesystem::exists(manifest_path)) {
        throw std::runtime_error("manifest already exists (pass overwrite=true to replace): " +
                                 manifest_path.string());
    }
    std::filesystem::create_directories(columns_dir);

    Reader reader;
    if (!reader.mmap(opts.input_csv.string())) {
        throw std::runtime_error("cannot open input csv: " + opts.input_csv.string());
    }

    auto it  = reader.begin();
    auto end = reader.end();
    // csv2's RowIterator exposes only operator!= and operator++; check
    // emptiness through the negated inequality.
    if (!(it != end)) throw std::runtime_error("input csv is empty");

    // Slurp one row's cells into a vector<string>. csv2's read_value
    // handles RFC-4180 quote escaping for us.
    auto slurp_row = [](auto&& row, std::vector<std::string>& out) {
        out.clear();
        std::string v;
        for (const auto cell : row) {
            v.clear();
            cell.read_value(v);
            out.push_back(std::move(v));
        }
    };

    std::vector<std::string> first_row;
    slurp_row(*it, first_row);

    std::vector<std::string> column_names;
    bool process_first_as_data = false;
    if (opts.has_header) {
        column_names = first_row;
        ++it;
    } else {
        column_names = duckdb_default_column_names(first_row.size());
        process_first_as_data = true;
    }

    std::unordered_map<std::string, std::uint32_t> name_to_index;
    name_to_index.reserve(column_names.size());
    for (std::uint32_t i = 0; i < column_names.size(); ++i) {
        name_to_index.emplace(column_names[i], i);
    }
    auto resolve = [&](const std::string& name) {
        const auto it_n = name_to_index.find(name);
        if (it_n == name_to_index.end()) {
            std::ostringstream msg;
            msg << "column name not found: '" << name << "'. available: ";
            for (std::size_t i = 0; i < column_names.size(); ++i) {
                if (i) msg << ", ";
                msg << column_names[i];
            }
            throw std::invalid_argument(msg.str());
        }
        return it_n->second;
    };

    std::vector<std::uint32_t> dim_src(opts.dimensions.size());
    for (std::size_t i = 0; i < opts.dimensions.size(); ++i) {
        dim_src[i] = resolve(opts.dimensions[i].name);
    }
    std::vector<std::uint32_t> meas_src(opts.measures.size());
    for (std::size_t i = 0; i < opts.measures.size(); ++i) {
        meas_src[i] = resolve(opts.measures[i]);
    }
    struct ResolvedFilter { std::uint32_t src; ValidationFilter::Op op; double value; };
    std::vector<ResolvedFilter> filters;
    filters.reserve(opts.validation_filters.size());
    for (const auto& f : opts.validation_filters) {
        filters.push_back({resolve(f.name), f.op, f.value});
    }

    std::vector<std::unique_ptr<ColumnSink>> dim_sinks;
    std::vector<std::unique_ptr<ColumnSink>> meas_sinks;
    std::vector<std::string> dim_file_rel(opts.dimensions.size());
    std::vector<std::string> meas_file_rel(opts.measures.size());
    for (std::size_t i = 0; i < opts.dimensions.size(); ++i) {
        dim_file_rel[i] = "columns/dim_" + std::to_string(i) + ".bin";
        dim_sinks.push_back(std::make_unique<ColumnSink>(opts.output_dir / dim_file_rel[i]));
    }
    for (std::size_t i = 0; i < opts.measures.size(); ++i) {
        meas_file_rel[i] = "columns/measure_" + std::to_string(i) + ".bin";
        meas_sinks.push_back(std::make_unique<ColumnSink>(opts.output_dir / meas_file_rel[i]));
    }

    std::vector<DimRunning>  dim_stats(opts.dimensions.size());
    std::vector<MeasRunning> meas_stats(opts.measures.size());

    ConvertReport rep;
    rep.manifest_path = manifest_path;

    std::vector<double> dim_vals(opts.dimensions.size());
    std::vector<double> meas_vals(opts.measures.size());

    const std::size_t expected_cols = column_names.size();
    const bool have_cap = opts.max_rows.has_value();
    const std::uint64_t cap = have_cap ? *opts.max_rows : 0;

    auto cell_double = [&](const std::vector<std::string>& cells, std::uint32_t src) -> double {
        const std::string& s = cells[src];
        if (s.empty()) return std::numeric_limits<double>::quiet_NaN();
        if (!opts.null_string.empty() && s == opts.null_string) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double v;
        if (!parse_double(s, v)) return std::numeric_limits<double>::quiet_NaN();
        return v;
    };

    // Returns false when the row cap is hit and we should stop iterating.
    auto process_row = [&](std::vector<std::string>& cells) -> bool {
        rep.rows_read++;
        // csv2 omits a trailing empty field after a terminal delimiter
        // ("a,b,\n" → 2 cells). Pad missing trailing fields back in so
        // we don't drop rows that simply have empty rightmost cells.
        if (cells.size() < expected_cols) {
            cells.resize(expected_cols);
        }

        for (const auto& f : filters) {
            if (!keep_under(f.op, cell_double(cells, f.src), f.value)) {
                rep.rows_filtered_out++;
                return true;
            }
        }

        // A point with an unknown coordinate has no place in the index.
        for (std::size_t i = 0; i < opts.dimensions.size(); ++i) {
            const double v = cell_double(cells, dim_src[i]);
            if (std::isnan(v)) { rep.rows_filtered_out++; return true; }
            dim_vals[i] = v;
        }
        for (std::size_t i = 0; i < opts.measures.size(); ++i) {
            meas_vals[i] = cell_double(cells, meas_src[i]);
        }

        for (std::size_t i = 0; i < opts.dimensions.size(); ++i) {
            dim_sinks[i]->write(dim_vals[i]);
            auto& s = dim_stats[i];
            if (dim_vals[i] < s.minv) s.minv = dim_vals[i];
            if (dim_vals[i] > s.maxv) s.maxv = dim_vals[i];
        }
        for (std::size_t i = 0; i < opts.measures.size(); ++i) {
            const double v = meas_vals[i];
            meas_sinks[i]->write(v);
            auto& s = meas_stats[i];
            if (std::isnan(v)) {
                s.nan_n++;
            } else {
                s.non_nan++;
                s.sum    += v;
                s.sum_sq += v * v;
                if (v < s.minv) s.minv = v;
                if (v > s.maxv) s.maxv = v;
            }
        }

        rep.rows_written++;
        return !(have_cap && rep.rows_written >= cap);
    };

    bool keep_going = true;
    if (process_first_as_data) {
        keep_going = process_row(first_row);
    }
    if (keep_going) {
        std::vector<std::string> row_cells;
        for (; it != end; ++it) {
            slurp_row(*it, row_cells);
            // csv2 can yield an empty trailing row for files ending in \n.
            if (row_cells.empty() || (row_cells.size() == 1 && row_cells[0].empty())) {
                continue;
            }
            if (!process_row(row_cells)) break;
        }
    }

    dim_sinks.clear();
    meas_sinks.clear();

    Manifest m;
    m.dataset_id = opts.dataset_id;
    m.row_count  = rep.rows_written;
    m.endianness = "little";
    m.dtype      = "float64";
    m.dimensions.reserve(opts.dimensions.size());
    for (std::uint16_t i = 0; i < opts.dimensions.size(); ++i) {
        DimensionDescriptor d;
        d.logical_id   = i;
        d.name         = opts.dimensions[i].name;
        d.source_index = dim_src[i];
        d.file         = dim_file_rel[i];
        d.min = std::isinf(dim_stats[i].minv) ? opts.dimensions[i].low  : dim_stats[i].minv;
        d.max = std::isinf(dim_stats[i].maxv) ? opts.dimensions[i].high : dim_stats[i].maxv;
        m.dimensions.push_back(std::move(d));
    }
    m.measures.reserve(opts.measures.size());
    for (std::uint16_t i = 0; i < opts.measures.size(); ++i) {
        MeasureDescriptor mz;
        mz.logical_id   = i;
        mz.name         = opts.measures[i];
        mz.source_index = meas_src[i];
        mz.file         = meas_file_rel[i];
        const auto& s = meas_stats[i];
        mz.global.non_nan_count = s.non_nan;
        mz.global.nan_count     = s.nan_n;
        mz.global.sum    = s.sum;
        mz.global.sum_sq = s.sum_sq;
        mz.global.min    = std::isinf(s.minv) ? 0.0 : s.minv;
        mz.global.max    = std::isinf(s.maxv) ? 0.0 : s.maxv;
        m.measures.push_back(std::move(mz));
    }
    m.domain_bounds.dims.reserve(opts.dimensions.size());
    for (const auto& dr : opts.dimensions) {
        m.domain_bounds.dims.push_back(Range{dr.low, dr.high});
    }
    m.null_encoding     = "NaN";
    m.source_file       = std::filesystem::absolute(opts.input_csv).string();
    m.source_sha256     = "";
    if (opts.max_rows) m.source_prefix_rows = *opts.max_rows;
    m.converter_version = opts.converter_version;
    m.created_utc       = utc_iso8601_now();

    write_manifest(manifest_path, m);
    return rep;
}

}  // namespace

std::vector<std::string> duckdb_default_column_names(std::size_t num_columns) {
    if (num_columns == 0) return {};
    const std::size_t max_idx = num_columns - 1;
    const std::size_t width = std::to_string(max_idx).size();
    std::vector<std::string> names;
    names.reserve(num_columns);
    for (std::size_t i = 0; i < num_columns; ++i) {
        auto s = std::to_string(i);
        if (s.size() < width) s.insert(0, width - s.size(), '0');
        names.push_back("column" + s);
    }
    return names;
}

ConvertReport run_csv_to_columns(const ConvertOptions& opts) {
    if (opts.input_csv.empty())  throw std::invalid_argument("input_csv is required");
    if (opts.output_dir.empty()) throw std::invalid_argument("output_dir is required");
    if (opts.dataset_id.empty()) throw std::invalid_argument("dataset_id is required");
    if (opts.dimensions.empty()) throw std::invalid_argument("at least one dimension is required");
    if (opts.measures.empty())   throw std::invalid_argument("at least one measure is required");

    // csv2 takes the delimiter as a template parameter; dispatch here on
    // the supported runtime values. Add more cases as new datasets need.
    switch (opts.delimiter) {
        case ',':  return run_with_delim<csv2::delimiter<','>>(opts);
        case '\t': return run_with_delim<csv2::delimiter<'\t'>>(opts);
        default:
            throw std::invalid_argument(
                std::string("unsupported delimiter: '") + opts.delimiter +
                "' (only ',' and '\\t' are supported)");
    }
}

}  // namespace a3i

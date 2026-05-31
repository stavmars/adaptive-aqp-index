#include "a3i/tools/parquet_to_columns_pipeline.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
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
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include "a3i/core/geometry.hpp"
#include "a3i/storage/manifest.hpp"

namespace a3i {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

template <class T>
T unwrap(arrow::Result<T> result, const std::string& what) {
    if (!result.ok()) {
        throw std::runtime_error(what + ": " + result.status().ToString());
    }
    return std::move(result).ValueUnsafe();
}

// One source column flattened to doubles with an explicit null mask. Null
// entries are stored as NaN; the mask lets us distinguish "absent" (drop a
// dimension row, NaN a measure) from a real value.
struct DoubleColumn {
    std::vector<double> value;
    std::vector<char>   is_null;
};

bool parse_double(std::string_view sv, double& out) {
    if (sv.empty()) return false;
    const auto res = std::from_chars(sv.data(), sv.data() + sv.size(), out);
    return res.ec == std::errc{} && res.ptr == sv.data() + sv.size();
}

// Flatten an Arrow column (any chunking) to doubles. Integers, floats and
// booleans convert numerically; strings are parsed (unparseable -> null);
// nulls are recorded in the mask. Unsupported types are a fatal error.
DoubleColumn materialize_double(const std::shared_ptr<arrow::ChunkedArray>& column,
                                const std::string& name) {
    DoubleColumn col;
    const auto n = static_cast<std::size_t>(column->length());
    col.value.reserve(n);
    col.is_null.reserve(n);

    auto push_null = [&] { col.value.push_back(kNaN); col.is_null.push_back(1); };
    auto push_val  = [&](double v) { col.value.push_back(v); col.is_null.push_back(0); };

    for (const auto& chunk : column->chunks()) {
        const auto tid = chunk->type_id();
        const std::int64_t len = chunk->length();
        switch (tid) {
            case arrow::Type::DOUBLE: {
                auto a = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                for (std::int64_t i = 0; i < len; ++i)
                    a->IsNull(i) ? push_null() : push_val(a->Value(i));
                break;
            }
            case arrow::Type::FLOAT: {
                auto a = std::static_pointer_cast<arrow::FloatArray>(chunk);
                for (std::int64_t i = 0; i < len; ++i)
                    a->IsNull(i) ? push_null() : push_val(static_cast<double>(a->Value(i)));
                break;
            }
            case arrow::Type::INT64: {
                auto a = std::static_pointer_cast<arrow::Int64Array>(chunk);
                for (std::int64_t i = 0; i < len; ++i)
                    a->IsNull(i) ? push_null() : push_val(static_cast<double>(a->Value(i)));
                break;
            }
            case arrow::Type::INT32: {
                auto a = std::static_pointer_cast<arrow::Int32Array>(chunk);
                for (std::int64_t i = 0; i < len; ++i)
                    a->IsNull(i) ? push_null() : push_val(static_cast<double>(a->Value(i)));
                break;
            }
            case arrow::Type::BOOL: {
                auto a = std::static_pointer_cast<arrow::BooleanArray>(chunk);
                for (std::int64_t i = 0; i < len; ++i)
                    a->IsNull(i) ? push_null() : push_val(a->Value(i) ? 1.0 : 0.0);
                break;
            }
            case arrow::Type::STRING: case arrow::Type::LARGE_STRING: {
                auto a = std::static_pointer_cast<arrow::StringArray>(chunk);
                for (std::int64_t i = 0; i < len; ++i) {
                    if (a->IsNull(i)) { push_null(); continue; }
                    const auto sv = a->GetView(i);
                    double v;
                    parse_double(std::string_view(sv.data(), sv.size()), v)
                        ? push_val(v) : push_null();
                }
                break;
            }
            default:
                throw std::invalid_argument(
                    "column '" + name + "' has unsupported type " +
                    chunk->type()->ToString());
        }
    }
    return col;
}

bool keep_under(const ValidationFilter::Op op, double cell, double rhs) {
    if (std::isnan(cell)) return true;  // unknown values never drop on numeric grounds
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

}  // namespace

ConvertReport run_parquet_to_columns(const ConvertOptions& opts) {
    if (opts.input_parquet.empty()) throw std::invalid_argument("input_parquet is required");
    if (opts.output_dir.empty())    throw std::invalid_argument("output_dir is required");
    if (opts.dataset_id.empty())    throw std::invalid_argument("dataset_id is required");
    if (opts.dimensions.empty())    throw std::invalid_argument("at least one dimension is required");
    if (opts.measures.empty())      throw std::invalid_argument("at least one measure is required");

    const auto columns_dir   = opts.output_dir / "columns";
    const auto manifest_path = opts.output_dir / "manifest.json";
    if (!opts.overwrite && std::filesystem::exists(manifest_path)) {
        throw std::runtime_error("manifest already exists (pass overwrite=true to replace): " +
                                 manifest_path.string());
    }

    // --- Read the Parquet ------------------------------------------------
    auto input = unwrap(arrow::io::ReadableFile::Open(opts.input_parquet.string()),
                        "cannot open input parquet: " + opts.input_parquet.string());
    auto reader = unwrap(parquet::arrow::OpenFile(input, arrow::default_memory_pool()),
                         "cannot open parquet reader: " + opts.input_parquet.string());
    auto table = unwrap(reader->ReadTable(),
                        "cannot read parquet: " + opts.input_parquet.string());

    const auto& schema = table->schema();
    std::unordered_map<std::string, std::uint32_t> name_to_index;
    name_to_index.reserve(static_cast<std::size_t>(schema->num_fields()));
    for (int i = 0; i < schema->num_fields(); ++i) {
        name_to_index.emplace(schema->field(i)->name(), static_cast<std::uint32_t>(i));
    }
    auto resolve = [&](const std::string& name) {
        const auto it_n = name_to_index.find(name);
        if (it_n == name_to_index.end()) {
            std::ostringstream msg;
            msg << "column name not found: '" << name << "'. available: ";
            for (int i = 0; i < schema->num_fields(); ++i) {
                if (i) msg << ", ";
                msg << schema->field(i)->name();
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

    // Materialize only the columns we actually need, once.
    std::unordered_map<std::uint32_t, DoubleColumn> cols;
    auto need = [&](std::uint32_t idx) {
        if (cols.find(idx) == cols.end()) {
            cols.emplace(idx, materialize_double(table->column(static_cast<int>(idx)),
                                                 schema->field(static_cast<int>(idx))->name()));
        }
    };
    for (auto s : dim_src)  need(s);
    for (auto s : meas_src) need(s);
    for (const auto& f : filters) need(f.src);

    // --- Output sinks -----------------------------------------------------
    std::filesystem::create_directories(columns_dir);
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

    const std::uint64_t total_rows = static_cast<std::uint64_t>(table->num_rows());
    const bool have_cap = opts.max_rows.has_value();
    const std::uint64_t cap = have_cap ? *opts.max_rows : 0;

    std::vector<double> dim_vals(opts.dimensions.size());

    for (std::uint64_t r = 0; r < total_rows; ++r) {
        rep.rows_read++;

        // Drop filters: remove the row if any predicate matches.
        bool dropped = false;
        for (const auto& f : filters) {
            const auto& c = cols.at(f.src);
            const double cell = c.is_null[r] ? kNaN : c.value[r];
            if (!keep_under(f.op, cell, f.value)) { dropped = true; break; }
        }
        if (dropped) { rep.rows_filtered_out++; continue; }

        // Dimensions: a missing or out-of-bounds coordinate drops the row.
        bool bad_dim = false;
        for (std::size_t i = 0; i < opts.dimensions.size(); ++i) {
            const auto& c = cols.at(dim_src[i]);
            if (c.is_null[r]) { bad_dim = true; break; }
            const double v = c.value[r];
            if (v < opts.dimensions[i].low || v > opts.dimensions[i].high) {
                bad_dim = true; break;
            }
            dim_vals[i] = v;
        }
        if (bad_dim) { rep.rows_filtered_out++; continue; }

        for (std::size_t i = 0; i < opts.dimensions.size(); ++i) {
            dim_sinks[i]->write(dim_vals[i]);
            auto& s = dim_stats[i];
            if (dim_vals[i] < s.minv) s.minv = dim_vals[i];
            if (dim_vals[i] > s.maxv) s.maxv = dim_vals[i];
        }
        for (std::size_t i = 0; i < opts.measures.size(); ++i) {
            const auto& c = cols.at(meas_src[i]);
            const double v = c.is_null[r] ? kNaN : c.value[r];
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
        if (have_cap && rep.rows_written >= cap) break;
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
    m.source_file       = std::filesystem::absolute(opts.input_parquet).string();
    m.source_sha256     = "";
    if (opts.max_rows) m.source_prefix_rows = *opts.max_rows;
    m.converter_version = opts.converter_version;
    m.created_utc       = utc_iso8601_now();

    write_manifest(manifest_path, m);
    return rep;
}

}  // namespace a3i

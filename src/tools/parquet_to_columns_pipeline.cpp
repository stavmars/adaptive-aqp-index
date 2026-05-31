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
#include <numeric>
#include <set>
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

void check(const arrow::Status& status, const std::string& what) {
    if (!status.ok()) {
        throw std::runtime_error(what + ": " + status.ToString());
    }
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

// Append one Arrow array to a double column with an explicit null mask.
// Integers, floats and booleans convert numerically; strings are parsed
// (unparseable -> null); nulls are recorded in the mask. Unsupported types are
// a fatal error. Called once per needed column per streamed batch, so peak
// memory stays bounded by the batch size rather than the whole file.
void append_array_as_double(const arrow::Array& chunk, const std::string& name,
                            DoubleColumn& col) {
    auto push_null = [&] { col.value.push_back(kNaN); col.is_null.push_back(1); };
    auto push_val  = [&](double v) { col.value.push_back(v); col.is_null.push_back(0); };

    const std::int64_t len = chunk.length();
    switch (chunk.type_id()) {
        case arrow::Type::DOUBLE: {
            const auto& a = static_cast<const arrow::DoubleArray&>(chunk);
            for (std::int64_t i = 0; i < len; ++i)
                a.IsNull(i) ? push_null() : push_val(a.Value(i));
            break;
        }
        case arrow::Type::FLOAT: {
            const auto& a = static_cast<const arrow::FloatArray&>(chunk);
            for (std::int64_t i = 0; i < len; ++i)
                a.IsNull(i) ? push_null() : push_val(static_cast<double>(a.Value(i)));
            break;
        }
        case arrow::Type::INT64: {
            const auto& a = static_cast<const arrow::Int64Array&>(chunk);
            for (std::int64_t i = 0; i < len; ++i)
                a.IsNull(i) ? push_null() : push_val(static_cast<double>(a.Value(i)));
            break;
        }
        case arrow::Type::INT32: {
            const auto& a = static_cast<const arrow::Int32Array&>(chunk);
            for (std::int64_t i = 0; i < len; ++i)
                a.IsNull(i) ? push_null() : push_val(static_cast<double>(a.Value(i)));
            break;
        }
        case arrow::Type::BOOL: {
            const auto& a = static_cast<const arrow::BooleanArray&>(chunk);
            for (std::int64_t i = 0; i < len; ++i)
                a.IsNull(i) ? push_null() : push_val(a.Value(i) ? 1.0 : 0.0);
            break;
        }
        case arrow::Type::STRING: case arrow::Type::LARGE_STRING: {
            const auto& a = static_cast<const arrow::StringArray&>(chunk);
            for (std::int64_t i = 0; i < len; ++i) {
                if (a.IsNull(i)) { push_null(); continue; }
                const auto sv = a.GetView(i);
                double v;
                parse_double(std::string_view(sv.data(), sv.size()), v)
                    ? push_val(v) : push_null();
            }
            break;
        }
        default:
            throw std::invalid_argument(
                "column '" + name + "' has unsupported type " +
                chunk.type()->ToString());
    }
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

    // --- Open the Parquet (streamed by row-group batch, never fully read) ---
    auto input = unwrap(arrow::io::ReadableFile::Open(opts.input_parquet.string()),
                        "cannot open input parquet: " + opts.input_parquet.string());
    auto reader = unwrap(parquet::arrow::OpenFile(input, arrow::default_memory_pool()),
                         "cannot open parquet reader: " + opts.input_parquet.string());
    std::shared_ptr<arrow::Schema> schema;
    check(reader->GetSchema(&schema),
          "cannot read parquet schema: " + opts.input_parquet.string());

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

    // Read only the columns we actually use, as the union of dimension,
    // measure and filter sources, sorted and de-duplicated. The record-batch
    // reader yields these columns in this order, so a source index maps to a
    // batch position by lower_bound.
    std::vector<int> needed;
    {
        std::set<std::uint32_t> uniq;
        for (auto s : dim_src)  uniq.insert(s);
        for (auto s : meas_src) uniq.insert(s);
        for (const auto& f : filters) uniq.insert(f.src);
        needed.reserve(uniq.size());
        for (auto s : uniq) needed.push_back(static_cast<int>(s));
    }
    auto batch_pos = [&](std::uint32_t src) -> std::size_t {
        const auto it = std::lower_bound(needed.begin(), needed.end(),
                                         static_cast<int>(src));
        return static_cast<std::size_t>(it - needed.begin());
    };
    std::vector<std::size_t> dim_pos(dim_src.size());
    for (std::size_t i = 0; i < dim_src.size(); ++i) dim_pos[i] = batch_pos(dim_src[i]);
    std::vector<std::size_t> meas_pos(meas_src.size());
    for (std::size_t i = 0; i < meas_src.size(); ++i) meas_pos[i] = batch_pos(meas_src[i]);
    std::vector<std::size_t> filter_pos(filters.size());
    for (std::size_t i = 0; i < filters.size(); ++i) filter_pos[i] = batch_pos(filters[i].src);

    // --- Output sinks -----------------------------------------------------
    std::filesystem::create_directories(columns_dir);
    std::vector<std::unique_ptr<ColumnSink>> dim_sinks;
    std::vector<std::unique_ptr<ColumnSink>> meas_sinks;
    std::vector<std::string> dim_file_rel(opts.dimensions.size());
    std::vector<std::string> meas_file_rel(opts.measures.size());
    // Each column file is named after the column it holds, so a prepared
    // directory is self-describing on disk; the manifest still maps logical ids
    // to these paths. Forward slashes in the relative path are intentional and
    // portable; the store opens the file through the manifest's path field.
    for (std::size_t i = 0; i < opts.dimensions.size(); ++i) {
        dim_file_rel[i] = "columns/" + opts.dimensions[i].name + ".bin";
        dim_sinks.push_back(std::make_unique<ColumnSink>(opts.output_dir / dim_file_rel[i]));
    }
    for (std::size_t i = 0; i < opts.measures.size(); ++i) {
        meas_file_rel[i] = "columns/" + opts.measures[i] + ".bin";
        meas_sinks.push_back(std::make_unique<ColumnSink>(opts.output_dir / meas_file_rel[i]));
    }

    std::vector<DimRunning>  dim_stats(opts.dimensions.size());
    std::vector<MeasRunning> meas_stats(opts.measures.size());

    ConvertReport rep;
    rep.manifest_path = manifest_path;

    const bool have_cap = opts.max_rows.has_value();
    const std::uint64_t cap = have_cap ? *opts.max_rows : 0;

    // --- Stream the Parquet one row-group batch at a time -----------------
    std::vector<int> row_groups(static_cast<std::size_t>(reader->num_row_groups()));
    std::iota(row_groups.begin(), row_groups.end(), 0);
    auto batch_reader = unwrap(reader->GetRecordBatchReader(row_groups, needed),
                               "cannot stream parquet: " + opts.input_parquet.string());

    std::vector<DoubleColumn> bcols(needed.size());
    std::vector<double> dim_vals(opts.dimensions.size());
    bool cap_reached = false;

    while (!cap_reached) {
        std::shared_ptr<arrow::RecordBatch> batch;
        check(batch_reader->ReadNext(&batch),
              "cannot read parquet batch: " + opts.input_parquet.string());
        if (!batch) break;  // end of stream
        const std::int64_t bn = batch->num_rows();

        // Decode this batch's needed columns to doubles; buffers are reused
        // across batches so peak memory is one batch, not the whole file.
        for (std::size_t c = 0; c < needed.size(); ++c) {
            bcols[c].value.clear();
            bcols[c].is_null.clear();
            bcols[c].value.reserve(static_cast<std::size_t>(bn));
            bcols[c].is_null.reserve(static_cast<std::size_t>(bn));
            append_array_as_double(*batch->column(static_cast<int>(c)),
                                   schema->field(needed[c])->name(), bcols[c]);
        }

        for (std::int64_t r = 0; r < bn; ++r) {
            rep.rows_read++;

            // Drop filters: remove the row if any predicate matches.
            bool dropped = false;
            for (std::size_t k = 0; k < filters.size(); ++k) {
                const auto& c = bcols[filter_pos[k]];
                const double cell = c.is_null[r] ? kNaN : c.value[r];
                if (!keep_under(filters[k].op, cell, filters[k].value)) { dropped = true; break; }
            }
            if (dropped) { rep.rows_filtered_out++; continue; }

            // Dimensions: a missing or out-of-bounds coordinate drops the row.
            bool bad_dim = false;
            for (std::size_t i = 0; i < opts.dimensions.size(); ++i) {
                const auto& c = bcols[dim_pos[i]];
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
                const auto& c = bcols[meas_pos[i]];
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
            if (have_cap && rep.rows_written >= cap) { cap_reached = true; break; }
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

    auto op_token = [](ValidationFilter::Op op) -> const char* {
        switch (op) {
            case ValidationFilter::Op::Lt: return "<";
            case ValidationFilter::Op::Le: return "<=";
            case ValidationFilter::Op::Gt: return ">";
            case ValidationFilter::Op::Ge: return ">=";
            case ValidationFilter::Op::Eq: return "==";
            case ValidationFilter::Op::Ne: return "!=";
        }
        return "?";
    };
    m.applied_drop_if.reserve(opts.validation_filters.size());
    for (const auto& f : opts.validation_filters) {
        std::ostringstream oss;
        oss << f.name << ' ' << op_token(f.op) << ' ' << f.value;
        m.applied_drop_if.push_back(oss.str());
    }

    m.source_parquet = std::filesystem::absolute(opts.input_parquet).string();
    std::error_code ec;
    const auto sz = std::filesystem::file_size(opts.input_parquet, ec);
    m.source_bytes = ec ? 0 : static_cast<std::uint64_t>(sz);
    const auto wt = std::filesystem::last_write_time(opts.input_parquet, ec);
    if (ec) {
        m.source_mtime = 0;
    } else {
        // Store as Unix epoch seconds so the prepare script (and any reader)
        // can compare against a plain stat() mtime for the staleness check.
        const auto sys = std::chrono::clock_cast<std::chrono::system_clock>(wt);
        m.source_mtime =
            std::chrono::duration_cast<std::chrono::seconds>(sys.time_since_epoch()).count();
    }
    if (opts.max_rows) m.max_rows = *opts.max_rows;
    if (opts.parent_dataset_id) m.parent_dataset_id = *opts.parent_dataset_id;
    m.converter_version = opts.converter_version;
    m.created_utc       = utc_iso8601_now();

    write_manifest(manifest_path, m);
    return rep;
}

}  // namespace a3i

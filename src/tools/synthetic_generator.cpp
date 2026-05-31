#include "a3i/tools/synthetic_generator.hpp"

#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include "a3i/util/rng.hpp"

namespace a3i {

namespace {

void check(const arrow::Status& status, const std::string& what) {
    if (!status.ok()) {
        throw std::runtime_error(what + ": " + status.ToString());
    }
}

template <class T>
T unwrap(arrow::Result<T> result, const std::string& what) {
    if (!result.ok()) {
        throw std::runtime_error(what + ": " + result.status().ToString());
    }
    return std::move(result).ValueUnsafe();
}

// Draw one column's worth of values. Drawing is sequential from the shared
// generator so the whole dataset is a deterministic function of the seed.
std::vector<double> draw_column(const SyntheticColumnSpec& spec, std::uint64_t rows,
                                Rng& rng) {
    std::vector<double> out;
    out.reserve(rows);
    switch (spec.distribution) {
        case SyntheticDistribution::Uniform: {
            if (!(spec.param0 < spec.param1)) {
                throw std::invalid_argument("uniform column '" + spec.name +
                                            "' needs param0 < param1");
            }
            std::uniform_real_distribution<double> dist(spec.param0, spec.param1);
            for (std::uint64_t i = 0; i < rows; ++i) out.push_back(dist(rng));
            break;
        }
        case SyntheticDistribution::Normal: {
            std::normal_distribution<double> dist(spec.param0, spec.param1);
            for (std::uint64_t i = 0; i < rows; ++i) out.push_back(dist(rng));
            break;
        }
        case SyntheticDistribution::LogNormal: {
            std::lognormal_distribution<double> dist(spec.param0, spec.param1);
            for (std::uint64_t i = 0; i < rows; ++i) out.push_back(dist(rng));
            break;
        }
    }
    return out;
}

}  // namespace

std::uint64_t file_fingerprint(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file for fingerprint: " + path.string());
    std::uint64_t h = 1469598103934665603ull;  // FNV-1a offset basis
    char buf[1 << 16];
    while (in) {
        in.read(buf, sizeof buf);
        const std::streamsize n = in.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            h ^= static_cast<unsigned char>(buf[i]);
            h *= 1099511628211ull;  // FNV prime
        }
    }
    return h;
}

GeneratorReport generate_parquet(const GeneratorConfig& config,
                                 const std::filesystem::path& output_path,
                                 bool overwrite) {
    if (output_path.empty())   throw std::invalid_argument("output_path is required");
    if (config.columns.empty()) throw std::invalid_argument("at least one column is required");
    if (!overwrite && std::filesystem::exists(output_path)) {
        throw std::runtime_error("output already exists (pass overwrite to replace): " +
                                 output_path.string());
    }
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    Rng rng(config.seed);

    arrow::FieldVector            fields;
    arrow::ArrayVector            arrays;
    fields.reserve(config.columns.size());
    arrays.reserve(config.columns.size());

    for (const auto& spec : config.columns) {
        const auto values = draw_column(spec, config.rows, rng);
        arrow::DoubleBuilder builder;
        check(builder.AppendValues(values), "cannot build column " + spec.name);
        std::shared_ptr<arrow::Array> array;
        check(builder.Finish(&array), "cannot finalize column " + spec.name);
        fields.push_back(arrow::field(spec.name, arrow::float64()));
        arrays.push_back(std::move(array));
    }

    auto schema = arrow::schema(fields);
    auto table  = arrow::Table::Make(schema, arrays,
                                     static_cast<std::int64_t>(config.rows));

    auto out = unwrap(arrow::io::FileOutputStream::Open(output_path.string()),
                      "cannot open output parquet: " + output_path.string());

    // Identical pinned properties to the CSV converter so generated and
    // converted Parquet share a single, deterministic on-disk encoding.
    parquet::WriterProperties::Builder builder;
    builder.compression(parquet::Compression::UNCOMPRESSED);
    builder.disable_dictionary();
    builder.version(parquet::ParquetVersion::PARQUET_2_6);
    auto props = builder.build();

    const std::int64_t chunk = config.rows > 0
        ? static_cast<std::int64_t>(config.rows) : 1;
    check(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                     out, chunk, props),
          "cannot write parquet: " + output_path.string());
    check(out->Close(), "cannot close parquet output");

    GeneratorReport rep;
    rep.output_path = output_path;
    rep.rows        = config.rows;
    rep.fingerprint = file_fingerprint(output_path);
    return rep;
}

}  // namespace a3i

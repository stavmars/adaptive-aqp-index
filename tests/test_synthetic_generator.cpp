// Determinism, seed-sensitivity, and distribution checks for the seeded
// synthetic Parquet generator, plus confirmation that a generated file feeds
// the binary converter and yields reproducible global statistics.

#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string_view>

#include "a3i/storage/binary_column_store.hpp"
#include "a3i/tools/parquet_to_columns_pipeline.hpp"
#include "a3i/tools/synthetic_generator.hpp"

namespace fs = std::filesystem;
using namespace a3i;

namespace {

class TempDir {
   public:
    TempDir() {
        path_ = fs::temp_directory_path() /
                ("a3i_gen_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    fs::path operator/(std::string_view leaf) const { return path_ / leaf; }

   private:
    fs::path path_;
};

GeneratorConfig two_column_config(std::uint64_t seed, std::uint64_t rows) {
    GeneratorConfig cfg;
    cfg.seed = seed;
    cfg.rows = rows;
    cfg.columns = {
        {"x", SyntheticDistribution::Uniform, 0.0, 100.0},
        {"m", SyntheticDistribution::Normal, 50.0, 5.0},
    };
    return cfg;
}

}  // namespace

// Same seed + config -> byte-identical output (equal fingerprint).
TEST(SyntheticGenerator, IsDeterministic) {
    TempDir tmp;
    const auto cfg = two_column_config(42, 1000);

    const auto a = generate_parquet(cfg, tmp / "a.parquet", /*overwrite=*/true);
    const auto b = generate_parquet(cfg, tmp / "b.parquet", /*overwrite=*/true);

    EXPECT_EQ(a.rows, 1000u);
    EXPECT_EQ(b.rows, 1000u);
    EXPECT_EQ(a.fingerprint, b.fingerprint);
    EXPECT_EQ(fs::file_size(a.output_path), fs::file_size(b.output_path));
}

// A different seed changes the drawn values, hence the fingerprint.
TEST(SyntheticGenerator, SeedChangesOutput) {
    TempDir tmp;
    const auto a = generate_parquet(two_column_config(1, 1000), tmp / "a.parquet", true);
    const auto b = generate_parquet(two_column_config(2, 1000), tmp / "b.parquet", true);
    EXPECT_NE(a.fingerprint, b.fingerprint);
}

// A normal column's sample mean is close to its requested mean.
TEST(SyntheticGenerator, NormalColumnMatchesRequestedMean) {
    TempDir tmp;
    const auto cfg = two_column_config(7, 50000);
    const auto rep = generate_parquet(cfg, tmp / "g.parquet", true);

    auto infile = arrow::io::ReadableFile::Open(rep.output_path.string()).ValueOrDie();
    auto reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool()).ValueOrDie();
    auto table  = reader->ReadTable().ValueOrDie();

    // Column index 1 is "m" ~ Normal(50, 5).
    auto m = std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
    ASSERT_EQ(m->length(), 50000);
    double sum = 0.0;
    for (std::int64_t i = 0; i < m->length(); ++i) sum += m->Value(i);
    const double mean = sum / static_cast<double>(m->length());
    EXPECT_NEAR(mean, 50.0, 0.2);
}

// Writing onto an existing file requires the overwrite flag.
TEST(SyntheticGenerator, OverwriteGuard) {
    TempDir tmp;
    const auto cfg = two_column_config(3, 10);
    const auto out = tmp / "out.parquet";

    ASSERT_NO_THROW((void)generate_parquet(cfg, out, /*overwrite=*/false));
    EXPECT_THROW((void)generate_parquet(cfg, out, /*overwrite=*/false), std::runtime_error);
    EXPECT_NO_THROW((void)generate_parquet(cfg, out, /*overwrite=*/true));
}

// A generated file feeds the converter and produces reproducible stats.
TEST(SyntheticGenerator, FeedsConverterWithReproducibleStats) {
    TempDir tmp;
    const auto cfg = two_column_config(99, 2000);

    auto convert = [&](const fs::path& parquet, const std::string& id) {
        ConvertOptions opts;
        opts.input_parquet = parquet;
        opts.output_dir    = tmp / id;
        opts.dataset_id     = id;
        opts.dimensions     = {{"x", 0.0, 100.0}};
        opts.measures       = {"m"};
        opts.overwrite      = true;
        const auto rep = run_parquet_to_columns(opts);
        return BinaryColumnStore(rep.manifest_path).global_stats(0);
    };

    const auto r1 = generate_parquet(cfg, tmp / "g1.parquet", true);
    const auto r2 = generate_parquet(cfg, tmp / "g2.parquet", true);
    ASSERT_EQ(r1.fingerprint, r2.fingerprint);

    const auto s1 = convert(tmp / "g1.parquet", "run1");
    const auto s2 = convert(tmp / "g2.parquet", "run2");

    EXPECT_EQ(s1.non_nan_count, 2000u);
    EXPECT_EQ(s1.nan_count, 0u);
    EXPECT_EQ(s1.non_nan_count, s2.non_nan_count);
    EXPECT_DOUBLE_EQ(s1.sum, s2.sum);
    EXPECT_DOUBLE_EQ(s1.sum_sq, s2.sum_sq);
    EXPECT_DOUBLE_EQ(s1.min, s2.min);
    EXPECT_DOUBLE_EQ(s1.max, s2.max);
}

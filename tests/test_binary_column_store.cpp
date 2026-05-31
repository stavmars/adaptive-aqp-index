// Tests for the offline CSV->binary converter + BinaryColumnStore.
//
// Each test runs the pipeline against a tiny in-process fixture written
// to a temp dir, then opens the result through BinaryColumnStore and
// asserts on the round-tripped data, manifest metadata, and stats.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "a3i/storage/binary_column_store.hpp"
#include "a3i/storage/manifest.hpp"
#include "a3i/tools/parquet_to_columns_pipeline.hpp"
#include "a3i/tools/csv_to_parquet.hpp"

namespace fs = std::filesystem;

namespace {

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        std::mt19937_64    rng(rd());
        path_ = fs::temp_directory_path() /
                ("a3i_test_" + std::to_string(rng()));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const fs::path& path() const { return path_; }
    fs::path operator/(const std::string& s) const { return path_ / s; }

private:
    fs::path path_;
};

void write_text(const fs::path& p, std::string_view content) {
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(content.data(), static_cast<std::streamsize>(content.size()));
}

// Run the one-off CSV -> Parquet step so the converter (which now
// reads Parquet) can consume a tiny text fixture. Returns the Parquet path.
fs::path to_parquet(const fs::path& csv, bool has_header, char delimiter = ',',
                    const std::string& null_string = "") {
    fs::path pq = csv;
    pq.replace_extension(".parquet");
    a3i::CsvToParquetOptions po;
    po.input_path  = csv;
    po.output_path = pq;
    po.has_header  = has_header;
    po.delimiter   = delimiter;
    po.null_string = null_string;
    po.overwrite   = true;
    a3i::csv_to_parquet(po);
    return pq;
}

}  // namespace

// --- default_column_names -----------------------------------------------

TEST(DefaultColumnNames, PaddingByDigitWidth) {
    using a3i::default_column_names;
    {
        auto names = default_column_names(10);
        // 10 cols -> indices 0..9, single-digit width -> col0..col9
        ASSERT_EQ(names.size(), 10u);
        EXPECT_EQ(names.front(), "col0");
        EXPECT_EQ(names.back(),  "col9");
    }
    {
        auto names = default_column_names(18);
        // 18 cols -> indices 0..17, two-digit width -> col00..col17
        ASSERT_EQ(names.size(), 18u);
        EXPECT_EQ(names.front(), "col00");
        EXPECT_EQ(names.back(),  "col17");
    }
    {
        auto names = default_column_names(100);
        // 100 cols -> indices 0..99, two-digit width
        ASSERT_EQ(names.size(), 100u);
        EXPECT_EQ(names.front(), "col00");
        EXPECT_EQ(names.back(),  "col99");
    }
}

// --- Headered round-trip ------------------------------------------------

TEST(BinaryColumnStore, HeaderedRoundTrip) {
    TempDir tmp;
    const auto csv = tmp / "taxi.csv";
    write_text(csv,
        "pickup_lon,pickup_lat,fare_amount,passenger_count\n"
        "-73.95,40.78,12.5,1\n"
        "-73.96,40.79,8.0,2\n"
        "-73.97,40.80,15.25,3\n");

    a3i::ConvertOptions opts;
    opts.input_parquet = to_parquet(csv, /*has_header=*/true);
    opts.output_dir = tmp / "prep";
    opts.dataset_id = "taxi_tiny";
    opts.dimensions = {
        {"pickup_lon", -74.5, -73.5},
        {"pickup_lat",  40.5,  41.5},
    };
    opts.measures = {"fare_amount", "passenger_count"};

    const auto rep = a3i::run_parquet_to_columns(opts);
    EXPECT_EQ(rep.rows_written, 3u);
    EXPECT_EQ(rep.rows_read,    3u);
    EXPECT_EQ(rep.rows_filtered_out, 0u);

    a3i::BinaryColumnStore store(rep.manifest_path);
    EXPECT_EQ(store.row_count(), 3u);
    EXPECT_EQ(store.dimension_count(), 2u);
    EXPECT_EQ(store.measure_count(),   2u);

    auto dim0 = store.dimension_column(0);
    ASSERT_EQ(dim0.size(), 3u);
    EXPECT_DOUBLE_EQ(dim0[0], -73.95);
    EXPECT_DOUBLE_EQ(dim0[2], -73.97);

    EXPECT_DOUBLE_EQ(store.measure_value(0, 0), 12.5);
    EXPECT_DOUBLE_EQ(store.measure_value(2, 0), 15.25);
    EXPECT_DOUBLE_EQ(store.measure_value(1, 1), 2.0);

    const auto& stats = store.global_stats(0);
    EXPECT_EQ(stats.non_nan_count, 3u);
    EXPECT_EQ(stats.nan_count,     0u);
    EXPECT_DOUBLE_EQ(stats.sum, 35.75);
    EXPECT_DOUBLE_EQ(stats.min, 8.0);
    EXPECT_DOUBLE_EQ(stats.max, 15.25);
}

// --- Headerless synthesized positional column names ---------------------

TEST(BinaryColumnStore, HeaderlessUsesPositionalColumnNames) {
    TempDir tmp;
    const auto csv = tmp / "synth.csv";
    write_text(csv,
        "0.1,0.2,10\n"
        "0.3,0.4,20\n");

    a3i::ConvertOptions opts;
    opts.input_parquet = to_parquet(csv, /*has_header=*/false);
    opts.output_dir = tmp / "prep";
    opts.dataset_id = "synth_tiny";
    opts.dimensions = {{"col0", 0.0, 1.0}, {"col1", 0.0, 1.0}};
    opts.measures   = {"col2"};

    const auto rep = a3i::run_parquet_to_columns(opts);
    EXPECT_EQ(rep.rows_written, 2u);

    a3i::BinaryColumnStore store(rep.manifest_path);
    EXPECT_DOUBLE_EQ(store.measure_value(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(store.measure_value(1, 0), 20.0);

    // Manifest preserves the synthesized names + source indices.
    const auto& m = store.manifest();
    ASSERT_EQ(m.dimensions.size(), 2u);
    EXPECT_EQ(m.dimensions[0].name, "col0");
    EXPECT_EQ(m.dimensions[0].source_index, 0u);
    EXPECT_EQ(m.dimensions[1].name, "col1");
    EXPECT_EQ(m.dimensions[1].source_index, 1u);
    EXPECT_EQ(m.measures[0].name, "col2");
    EXPECT_EQ(m.measures[0].source_index, 2u);
}

// --- NaN preservation and global stats ----------------------------------

TEST(BinaryColumnStore, NullsBecomeNaNAndStatsExcludeThem) {
    TempDir tmp;
    const auto csv = tmp / "nulls.csv";
    write_text(csv,
        "x,y,m\n"
        "0,0,1.0\n"
        "1,1,NA\n"           // null_string sentinel
        "2,2,\n"             // empty field
        "3,3,3.0\n");

    a3i::ConvertOptions opts;
    opts.input_parquet = to_parquet(csv, /*has_header=*/true, ',', /*null_string=*/"NA");
    opts.output_dir  = tmp / "prep";
    opts.dataset_id  = "nulls";
    opts.dimensions  = {{"x", 0, 10}, {"y", 0, 10}};
    opts.measures    = {"m"};

    const auto rep = a3i::run_parquet_to_columns(opts);
    ASSERT_EQ(rep.rows_written, 4u);

    a3i::BinaryColumnStore store(rep.manifest_path);
    EXPECT_DOUBLE_EQ(store.measure_value(0, 0), 1.0);
    EXPECT_TRUE(std::isnan(store.measure_value(1, 0)));
    EXPECT_TRUE(std::isnan(store.measure_value(2, 0)));
    EXPECT_DOUBLE_EQ(store.measure_value(3, 0), 3.0);

    const auto& s = store.global_stats(0);
    EXPECT_EQ(s.non_nan_count, 2u);
    EXPECT_EQ(s.nan_count,     2u);
    EXPECT_DOUBLE_EQ(s.sum,    4.0);
    EXPECT_DOUBLE_EQ(s.sum_sq, 10.0);
    EXPECT_DOUBLE_EQ(s.min, 1.0);
    EXPECT_DOUBLE_EQ(s.max, 3.0);
}

// --- gather() preserves input order -------------------------------------

TEST(BinaryColumnStore, GatherReturnsValuesInInputOrder) {
    TempDir tmp;
    const auto csv = tmp / "g.csv";
    {
        std::string s = "x,m\n";
        for (int i = 0; i < 10; ++i) {
            s += std::to_string(i) + "," + std::to_string(i * 100) + "\n";
        }
        write_text(csv, s);
    }

    a3i::ConvertOptions opts;
    opts.input_parquet = to_parquet(csv, /*has_header=*/true);
    opts.output_dir = tmp / "prep";
    opts.dataset_id = "g";
    opts.dimensions = {{"x", 0, 100}};
    opts.measures   = {"m"};
    auto rep = a3i::run_parquet_to_columns(opts);
    ASSERT_EQ(rep.rows_written, 10u);

    a3i::BinaryColumnStore store(rep.manifest_path);

    // Deliberately unsorted ids, including a duplicate.
    std::array<a3i::RowId, 6> ids{7, 2, 9, 0, 5, 2};
    std::array<double, 6>     out{};
    store.gather(/*m=*/0, ids, out);
    EXPECT_DOUBLE_EQ(out[0], 700.0);
    EXPECT_DOUBLE_EQ(out[1], 200.0);
    EXPECT_DOUBLE_EQ(out[2], 900.0);
    EXPECT_DOUBLE_EQ(out[3],   0.0);
    EXPECT_DOUBLE_EQ(out[4], 500.0);
    EXPECT_DOUBLE_EQ(out[5], 200.0);
}

// --- Validation filters DROP rows; --max-rows caps after filtering ------

TEST(BinaryColumnStore, ValidationFiltersDropRows) {
    TempDir tmp;
    const auto csv = tmp / "f.csv";
    write_text(csv,
        "x,m,bad\n"
        "0,1,0\n"
        "1,2,1\n"     // dropped: bad==1
        "2,3,0\n"
        "3,4,1\n");   // dropped: bad==1

    a3i::ConvertOptions opts;
    opts.input_parquet = to_parquet(csv, /*has_header=*/true);
    opts.output_dir = tmp / "prep";
    opts.dataset_id = "f";
    opts.dimensions = {{"x", 0, 10}};
    opts.measures   = {"m"};
    opts.validation_filters = {{"bad", a3i::ValidationFilter::Op::Eq, 1.0}};

    auto rep = a3i::run_parquet_to_columns(opts);
    EXPECT_EQ(rep.rows_written,      2u);
    EXPECT_EQ(rep.rows_filtered_out, 2u);

    a3i::BinaryColumnStore store(rep.manifest_path);
    auto x = store.dimension_column(0);
    ASSERT_EQ(x.size(), 2u);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
    EXPECT_DOUBLE_EQ(x[1], 2.0);
}

TEST(BinaryColumnStore, MaxRowsCaps) {
    TempDir tmp;
    const auto csv = tmp / "cap.csv";
    {
        std::string s = "x,m\n";
        for (int i = 0; i < 20; ++i) {
            s += std::to_string(i) + "," + std::to_string(i) + "\n";
        }
        write_text(csv, s);
    }
    a3i::ConvertOptions opts;
    opts.input_parquet = to_parquet(csv, /*has_header=*/true);
    opts.output_dir = tmp / "prep";
    opts.dataset_id = "cap";
    opts.dimensions = {{"x", 0, 100}};
    opts.measures   = {"m"};
    opts.max_rows   = 5u;

    auto rep = a3i::run_parquet_to_columns(opts);
    EXPECT_EQ(rep.rows_written, 5u);

    a3i::BinaryColumnStore store(rep.manifest_path);
    EXPECT_EQ(store.row_count(), 5u);
    EXPECT_TRUE(store.manifest().max_rows.has_value());
    EXPECT_EQ(*store.manifest().max_rows, 5u);
}

// --- Idempotency / overwrite -------------------------------------------

TEST(BinaryColumnStore, ManifestExistsBlocksWithoutOverwrite) {
    TempDir tmp;
    const auto csv = tmp / "i.csv";
    write_text(csv, "x,m\n0,0\n1,1\n");

    a3i::ConvertOptions opts;
    opts.input_parquet = to_parquet(csv, /*has_header=*/true);
    opts.output_dir = tmp / "prep";
    opts.dataset_id = "i";
    opts.dimensions = {{"x", 0, 10}};
    opts.measures   = {"m"};

    ASSERT_NO_THROW((void)a3i::run_parquet_to_columns(opts));
    EXPECT_THROW((void)a3i::run_parquet_to_columns(opts), std::runtime_error);

    opts.overwrite = true;
    EXPECT_NO_THROW((void)a3i::run_parquet_to_columns(opts));
}

// --- Unknown name lists available names --------------------------------

TEST(BinaryColumnStore, UnknownNameThrowsWithAvailableList) {
    TempDir tmp;
    const auto csv = tmp / "n.csv";
    write_text(csv, "alpha,beta,gamma\n1,2,3\n");

    a3i::ConvertOptions opts;
    opts.input_parquet = to_parquet(csv, /*has_header=*/true);
    opts.output_dir = tmp / "prep";
    opts.dataset_id = "n";
    opts.dimensions = {{"oops", 0, 10}};
    opts.measures   = {"gamma"};

    try {
        (void)a3i::run_parquet_to_columns(opts);
        FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("oops"),  std::string::npos);
        EXPECT_NE(what.find("alpha"), std::string::npos);
        EXPECT_NE(what.find("gamma"), std::string::npos);
    }
}

// --- Manifest schema discipline -----------------------------------------

TEST(BinaryColumnStore, ManifestLogicalIdEqualsPositionAndFilesAreRelative) {
    TempDir tmp;
    const auto csv = tmp / "s.csv";
    write_text(csv, "a,b,c,d\n1,2,3,4\n5,6,7,8\n");

    a3i::ConvertOptions opts;
    opts.input_parquet = to_parquet(csv, /*has_header=*/true);
    opts.output_dir = tmp / "prep";
    opts.dataset_id = "s";
    opts.dimensions = {{"a", 0, 10}, {"b", 0, 10}};
    opts.measures   = {"c", "d"};

    auto rep = a3i::run_parquet_to_columns(opts);
    auto m = a3i::read_manifest(rep.manifest_path);

    for (std::uint16_t i = 0; i < m.dimensions.size(); ++i) {
        EXPECT_EQ(m.dimensions[i].logical_id, i);
        EXPECT_FALSE(fs::path(m.dimensions[i].file).is_absolute());
    }
    for (std::uint16_t i = 0; i < m.measures.size(); ++i) {
        EXPECT_EQ(m.measures[i].logical_id, i);
        EXPECT_FALSE(fs::path(m.measures[i].file).is_absolute());
    }
    EXPECT_EQ(m.endianness, "little");
    EXPECT_EQ(m.dtype,      "float64");
    EXPECT_EQ(m.null_encoding, "NaN");
}

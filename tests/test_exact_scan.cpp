// Tests for the exact-scan ground-truth oracle.
//
// Each test builds a tiny prepared dataset via the offline converter, opens
// it through BinaryColumnStore, and checks exact_scan against hand-computed
// aggregates for a battery of rectangles (subset, full domain, empty,
// single row, all-NaN measure).

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "a3i/core/query.hpp"
#include "a3i/core/schema.hpp"
#include "a3i/experiments/truth_store.hpp"
#include "a3i/storage/binary_column_store.hpp"
#include "a3i/tools/parquet_to_columns_pipeline.hpp"
#include "a3i/tools/csv_to_parquet.hpp"

namespace fs = std::filesystem;

namespace {

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        std::mt19937_64    rng(rd());
        path_ = fs::temp_directory_path() / ("a3i_exact_" + std::to_string(rng()));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    fs::path operator/(const std::string& s) const { return path_ / s; }

private:
    fs::path path_;
};

// Four 2D points with two measures; the second measure of row 1 is missing
// (empty cell -> NaN). Domain is [0,3) x [0,3).
//   row x    y    m0  m1
//   0   0.5  0.5  10  1
//   1   1.5  0.5  20  (NaN)
//   2   0.5  1.5  30  3
//   3   2.5  2.5  40  4
constexpr const char* kCsv =
    "x,y,m0,m1\n"
    "0.5,0.5,10,1\n"
    "1.5,0.5,20,\n"
    "0.5,1.5,30,3\n"
    "2.5,2.5,40,4\n";

struct Fixture {
    TempDir                            tmp;
    std::optional<a3i::BinaryColumnStore> store;
    a3i::DatasetSchema                 schema;

    Fixture() {
        const auto csv = tmp / "data.csv";
        {
            std::ofstream o(csv, std::ios::binary | std::ios::trunc);
            o << kCsv;
        }
        const auto parquet = tmp / "data.parquet";
        {
            a3i::CsvToParquetOptions po;
            po.input_path  = csv;
            po.output_path = parquet;
            po.has_header  = true;
            po.delimiter   = ',';
            po.overwrite   = true;
            a3i::csv_to_parquet(po);
        }
        a3i::ConvertOptions opts;
        opts.input_parquet = parquet;
        opts.output_dir = tmp / "prepared";
        opts.dataset_id = "fixture";
        opts.dimensions = {{"x", 0.0, 3.0}, {"y", 0.0, 3.0}};
        opts.measures   = {"m0", "m1"};
        opts.overwrite  = true;
        const auto report = a3i::run_parquet_to_columns(opts);

        store.emplace(report.manifest_path);
        schema.dimension_names    = {"x", "y"};
        schema.measure_names      = {"m0", "m1"};
        schema.domain_bounds      = a3i::HyperRect{{{0.0, 3.0}, {0.0, 3.0}}};
        schema.object_count       = store->row_count();
        schema.binary_manifest_path = report.manifest_path.string();
    }
};

a3i::RangeQuery query(double xlo, double xhi, double ylo, double yhi) {
    return a3i::RangeQuery{a3i::HyperRect{{{xlo, xhi}, {ylo, yhi}}}, {0.0, 0.95}};
}

const a3i::AggregateEstimate& find(const a3i::QueryResult& r,
                                   a3i::AggregateOp op,
                                   a3i::MeasureId   mid) {
    for (const auto& e : r.aggregates) {
        if (e.op == op && (op == a3i::AggregateOp::CountStar || e.measure_id == mid)) {
            return e;
        }
    }
    ADD_FAILURE() << "aggregate not found";
    static a3i::AggregateEstimate dummy;
    return dummy;
}

void expect_exact(const a3i::AggregateEstimate& e, double value) {
    EXPECT_TRUE(e.exact);
    EXPECT_DOUBLE_EQ(e.estimate, value);
    EXPECT_DOUBLE_EQ(e.ci_low, value);
    EXPECT_DOUBLE_EQ(e.ci_high, value);
    EXPECT_DOUBLE_EQ(e.relative_half_width, 0.0);
}

}  // namespace

TEST(ExactScan, ShapeAllExact) {
    Fixture f;
    auto r = exact_scan(*f.store, f.schema, query(0.0, 3.0, 0.0, 3.0));
    // 3 estimates per measure + one COUNT(*).
    ASSERT_EQ(r.aggregates.size(), 3u * 2u + 1u);
    for (const auto& e : r.aggregates) EXPECT_TRUE(e.exact);
}

TEST(ExactScan, SubsetRectangle) {
    Fixture f;
    // [0,2) x [0,2) -> rows 0,1,2 qualify (row 3 at (2.5,2.5) excluded).
    auto r = exact_scan(*f.store, f.schema, query(0.0, 2.0, 0.0, 2.0));
    using a3i::AggregateOp;
    expect_exact(find(r, AggregateOp::CountStar, 0), 3.0);
    // m0: 10 + 20 + 30 = 60, count 3, avg 20.
    expect_exact(find(r, AggregateOp::Sum, 0), 60.0);
    expect_exact(find(r, AggregateOp::CountMeasure, 0), 3.0);
    expect_exact(find(r, AggregateOp::Avg, 0), 20.0);
    // m1: 1 + (NaN) + 3 = 4, count 2, avg 2.
    expect_exact(find(r, AggregateOp::Sum, 1), 4.0);
    expect_exact(find(r, AggregateOp::CountMeasure, 1), 2.0);
    expect_exact(find(r, AggregateOp::Avg, 1), 2.0);
}

TEST(ExactScan, FullDomain) {
    Fixture f;
    auto r = exact_scan(*f.store, f.schema, query(0.0, 3.0, 0.0, 3.0));
    using a3i::AggregateOp;
    expect_exact(find(r, AggregateOp::CountStar, 0), 4.0);
    expect_exact(find(r, AggregateOp::Sum, 0), 100.0);
    expect_exact(find(r, AggregateOp::CountMeasure, 0), 4.0);
    expect_exact(find(r, AggregateOp::Avg, 0), 25.0);
    // m1: 1 + 3 + 4 = 8, count 3 (row 1 missing), avg 8/3.
    expect_exact(find(r, AggregateOp::Sum, 1), 8.0);
    expect_exact(find(r, AggregateOp::CountMeasure, 1), 3.0);
    expect_exact(find(r, AggregateOp::Avg, 1), 8.0 / 3.0);
}

TEST(ExactScan, EmptyRectangle) {
    Fixture f;
    auto r = exact_scan(*f.store, f.schema, query(10.0, 11.0, 10.0, 11.0));
    using a3i::AggregateOp;
    expect_exact(find(r, AggregateOp::CountStar, 0), 0.0);
    expect_exact(find(r, AggregateOp::Sum, 0), 0.0);
    expect_exact(find(r, AggregateOp::CountMeasure, 0), 0.0);
    const auto& avg = find(r, AggregateOp::Avg, 0);
    EXPECT_TRUE(avg.exact);
    EXPECT_TRUE(std::isnan(avg.estimate));  // 0 count -> NaN
}

TEST(ExactScan, SingleRowWithMissingMeasure) {
    Fixture f;
    // [1,2) x [0,1) -> only row 1 (1.5,0.5); its m1 is NaN.
    auto r = exact_scan(*f.store, f.schema, query(1.0, 2.0, 0.0, 1.0));
    using a3i::AggregateOp;
    expect_exact(find(r, AggregateOp::CountStar, 0), 1.0);
    expect_exact(find(r, AggregateOp::Sum, 0), 20.0);
    expect_exact(find(r, AggregateOp::CountMeasure, 0), 1.0);
    expect_exact(find(r, AggregateOp::Avg, 0), 20.0);
    // m1 missing for the only qualifying row.
    expect_exact(find(r, AggregateOp::Sum, 1), 0.0);
    expect_exact(find(r, AggregateOp::CountMeasure, 1), 0.0);
    EXPECT_TRUE(std::isnan(find(r, AggregateOp::Avg, 1).estimate));
}

TEST(ExactScan, RejectsWrongDimensionality) {
    Fixture f;
    a3i::RangeQuery q{a3i::HyperRect{{{0.0, 1.0}}}, {0.0, 0.95}};
    EXPECT_THROW(exact_scan(*f.store, f.schema, q), std::invalid_argument);
}

TEST(ExactScan, MetricsReportFullScan) {
    Fixture f;
    auto r = exact_scan(*f.store, f.schema, query(0.0, 2.0, 0.0, 2.0));
    EXPECT_EQ(r.metrics.method, "exact_scan");
    EXPECT_EQ(r.metrics.status, "exact");
    EXPECT_TRUE(r.metrics.target_satisfied);
    EXPECT_EQ(r.metrics.rows_examined, f.store->row_count());
    EXPECT_EQ(r.metrics.measure_reads, 3u * 2u);  // 3 qualifying rows x 2 measures
}

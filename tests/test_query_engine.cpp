// The accuracy-aware query engine: exact mode reproduces the oracle, the
// approximate loop certifies its intervals, and a tight target falls back to
// a full read.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "a3i/core/query.hpp"
#include "a3i/core/schema.hpp"
#include "a3i/engine/query_engine.hpp"
#include "a3i/experiments/truth_store.hpp"
#include "a3i/storage/binary_column_store.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/adaptive_kd_access_path.hpp"
#include "a3i/tools/csv_to_columns_pipeline.hpp"

namespace fs = std::filesystem;

namespace {

using namespace a3i;

constexpr int kGrid = 20;  // 20 x 20 = 400 rows

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        std::mt19937_64    rng(rd());
        path_ = fs::temp_directory_path() / ("a3i_engine_" + std::to_string(rng()));
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

// A grid dataset with two measures. Row r = x * kGrid + y for x, y in
// [0, kGrid). m0 is always present (x * 100 + y); m1 is present only where
// x < 15, so the far corner has an entirely-missing measure.
struct Fixture {
    TempDir                               tmp;
    std::optional<BinaryColumnStore>      store;
    DatasetSchema                         schema;
    std::vector<double>                   xs, ys;

    Fixture() {
        const auto csv = tmp / "data.csv";
        {
            std::ofstream o(csv, std::ios::binary | std::ios::trunc);
            o << "x,y,m0,m1\n";
            for (int x = 0; x < kGrid; ++x) {
                for (int y = 0; y < kGrid; ++y) {
                    xs.push_back(static_cast<double>(x));
                    ys.push_back(static_cast<double>(y));
                    o << x << ',' << y << ',' << (x * 100 + y) << ',';
                    if (x < 15) o << (x + y);
                    o << '\n';
                }
            }
        }
        ConvertOptions opts;
        opts.input_csv  = csv;
        opts.output_dir = tmp / "prepared";
        opts.dataset_id = "engine_fixture";
        opts.has_header = true;
        opts.delimiter  = ',';
        opts.dimensions = {{"x", 0.0, kGrid}, {"y", 0.0, kGrid}};
        opts.measures   = {"m0", "m1"};
        opts.overwrite  = true;
        const auto report = run_csv_to_columns(opts);

        store.emplace(report.manifest_path);
        schema.dimension_names      = {"x", "y"};
        schema.measure_names        = {"m0", "m1"};
        schema.domain_bounds        = HyperRect{{{0.0, kGrid}, {0.0, kGrid}}};
        schema.object_count         = store->row_count();
        schema.binary_manifest_path = report.manifest_path.string();
    }

    IndexTable make_table() const { return IndexTable::from_columns({xs, ys}); }

    SubstrateConfig substrate() const {
        SubstrateConfig cfg;
        cfg.domain_bounds        = HyperRect{{{0.0, kGrid}, {0.0, kGrid}}};
        cfg.refinement_threshold = 16;
        return cfg;
    }
};

RangeQuery query(double xlo, double xhi, double ylo, double yhi, double rel) {
    return RangeQuery{HyperRect{{{xlo, xhi}, {ylo, yhi}}}, {rel, 0.95}};
}

const AggregateEstimate& find(const QueryResult& r, AggregateOp op, MeasureId mid) {
    for (const auto& e : r.aggregates) {
        if (e.op == op && (op == AggregateOp::CountStar || e.measure_id == mid)) {
            return e;
        }
    }
    ADD_FAILURE() << "aggregate not found";
    static AggregateEstimate dummy;
    return dummy;
}

void expect_matches_oracle(const QueryResult& got, const QueryResult& oracle) {
    ASSERT_EQ(got.aggregates.size(), oracle.aggregates.size());
    for (std::size_t i = 0; i < got.aggregates.size(); ++i) {
        const auto& e = got.aggregates[i];
        const auto& o = oracle.aggregates[i];
        EXPECT_EQ(static_cast<int>(e.op), static_cast<int>(o.op));
        EXPECT_EQ(e.measure_id, o.measure_id);
        EXPECT_TRUE(e.exact);
        const bool is_count = e.op == AggregateOp::CountMeasure ||
                              e.op == AggregateOp::CountStar;
        if (std::isnan(o.estimate)) {
            EXPECT_TRUE(std::isnan(e.estimate)) << "aggregate " << i;
        } else if (is_count) {
            EXPECT_DOUBLE_EQ(e.estimate, o.estimate) << "aggregate " << i;
            EXPECT_DOUBLE_EQ(e.ci_low, o.estimate);
            EXPECT_DOUBLE_EQ(e.ci_high, o.estimate);
        } else {
            const double tol = 1e-6 * std::max(1.0, std::abs(o.estimate));
            EXPECT_NEAR(e.estimate, o.estimate, tol) << "aggregate " << i;
            EXPECT_NEAR(e.ci_low, o.estimate, tol);
            EXPECT_NEAR(e.ci_high, o.estimate, tol);
        }
    }
}

}  // namespace

TEST(QueryEngine, ForceExactReproducesOracle) {
    Fixture f;
    IndexTable table = f.make_table();
    AdaptiveKdAccessPath path(f.substrate());
    path.prepare(table);

    EngineConfig cfg;
    cfg.accuracy_mode = EngineConfig::AccuracyMode::ForceExact;
    cfg.persist_summaries = false;
    QueryEngine engine(*f.store, table, path, cfg);

    const RangeQuery q = query(2.0, 12.0, 3.0, 17.0, /*rel=*/0.05);
    const QueryResult got = engine.execute(q, /*ordinal=*/1);
    const QueryResult oracle = exact_scan(*f.store, f.schema, q);

    expect_matches_oracle(got, oracle);
    EXPECT_EQ(got.metrics.status, "exact");
    EXPECT_EQ(got.metrics.exactify_cause, "none");
}

TEST(QueryEngine, ForceExactEmptyRectangle) {
    Fixture f;
    IndexTable table = f.make_table();
    AdaptiveKdAccessPath path(f.substrate());
    path.prepare(table);
    EngineConfig cfg;
    cfg.accuracy_mode = EngineConfig::AccuracyMode::ForceExact;
    QueryEngine engine(*f.store, table, path, cfg);

    const RangeQuery q = query(100.0, 110.0, 100.0, 110.0, 0.0);
    const QueryResult got = engine.execute(q, 1);
    const QueryResult oracle = exact_scan(*f.store, f.schema, q);
    expect_matches_oracle(got, oracle);
    expect_matches_oracle(got, oracle);
    EXPECT_DOUBLE_EQ(find(got, AggregateOp::CountStar, 0).estimate, 0.0);
}

TEST(QueryEngine, ForceExactAllMissingMeasure) {
    Fixture f;
    IndexTable table = f.make_table();
    AdaptiveKdAccessPath path(f.substrate());
    path.prepare(table);
    EngineConfig cfg;
    cfg.accuracy_mode = EngineConfig::AccuracyMode::ForceExact;
    QueryEngine engine(*f.store, table, path, cfg);

    // x in [16, 20): m1 is entirely missing there.
    const RangeQuery q = query(16.0, 20.0, 0.0, 20.0, 0.0);
    const QueryResult got = engine.execute(q, 1);
    const QueryResult oracle = exact_scan(*f.store, f.schema, q);
    expect_matches_oracle(got, oracle);
    EXPECT_DOUBLE_EQ(find(got, AggregateOp::CountMeasure, 1).estimate, 0.0);
    EXPECT_TRUE(std::isnan(find(got, AggregateOp::Avg, 1).estimate));
}

TEST(QueryEngine, ApproximateLoopCertifiesIntervals) {
    Fixture f;
    IndexTable table = f.make_table();
    AdaptiveKdAccessPath path(f.substrate());
    path.prepare(table);
    EngineConfig cfg;
    cfg.accuracy_mode = EngineConfig::AccuracyMode::PerQuery;
    QueryEngine engine(*f.store, table, path, cfg);

    const double rel = 0.25;
    const RangeQuery q = query(0.0, 20.0, 0.0, 20.0, rel);
    const QueryResult got = engine.execute(q, 7);
    const QueryResult oracle = exact_scan(*f.store, f.schema, q);

    EXPECT_TRUE(got.metrics.target_satisfied);
    for (const auto& e : got.aggregates) {
        if (e.exact) continue;
        EXPECT_LE(e.relative_half_width, rel) << static_cast<int>(e.op);
        EXPECT_LE(e.ci_low, e.estimate);
        EXPECT_LE(e.estimate, e.ci_high);
        EXPECT_GT(e.effective_df, 0.0);

        // The interval must not merely be self-consistent: the point estimate
        // has to land near the true value. With a fixed seed this is
        // deterministic, so a few half-widths of slack guards the calibration
        // without flaking.
        const auto& o = find(oracle, e.op, e.measure_id);
        if (std::isnan(o.estimate)) continue;
        const double half_width = 0.5 * (e.ci_high - e.ci_low);
        EXPECT_LE(std::abs(e.estimate - o.estimate), 4.0 * half_width)
            << "estimate far from truth for op " << static_cast<int>(e.op);
    }
    const std::string& s = got.metrics.status;
    EXPECT_TRUE(s == "converged" || s == "exactified");

}

TEST(QueryEngine, TightTargetFallsBackToExact) {
    Fixture f;
    IndexTable table = f.make_table();
    AdaptiveKdAccessPath path(f.substrate());
    path.prepare(table);
    EngineConfig cfg;
    cfg.accuracy_mode = EngineConfig::AccuracyMode::PerQuery;
    QueryEngine engine(*f.store, table, path, cfg);

    const RangeQuery q = query(0.0, 20.0, 0.0, 20.0, /*rel=*/1e-9);
    const QueryResult got = engine.execute(q, 3);
    const QueryResult oracle = exact_scan(*f.store, f.schema, q);

    expect_matches_oracle(got, oracle);
    EXPECT_EQ(got.metrics.status, "exactified");
    EXPECT_NE(got.metrics.exactify_cause, "none");
}

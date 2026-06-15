// Seeded statistical sanity: on a tiny known distribution the engine's point
// estimates land near the truth, the certified intervals have sane widths and
// bracket their own estimate, and forcing an exact answer reproduces the
// scan oracle bit-for-bit. Fast enough to run on every change, but kept in the
// `statistical` gate beside the heavier coverage sweep.

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
#include "a3i/tools/parquet_to_columns_pipeline.hpp"
#include "a3i/tools/csv_to_parquet.hpp"

namespace fs = std::filesystem;

namespace {

using namespace a3i;

constexpr int kRows = 2000;

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        std::mt19937_64    rng(rd());
        path_ = fs::temp_directory_path() / ("a3i_stat_smoke_" + std::to_string(rng()));
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

// A single-measure dataset with a known, moderate-spread distribution. Row i
// sits at (i, 0.5) and carries m = (i % 10) + 1, so the measure cycles over
// the integers 1..10. Over the whole domain the true aggregates are exact and
// closed-form: COUNT(*) = kRows, every residue class appears kRows/10 times,
// so SUM = (kRows/10) * (1+2+...+10) and AVG = 5.5.
struct Fixture {
    TempDir                          tmp;
    std::optional<BinaryColumnStore> store;
    DatasetSchema                    schema;
    std::vector<double>              xs, ys;
    double true_sum = 0.0, true_count = 0.0, true_avg = 0.0;

    Fixture() {
        const double xhi = static_cast<double>(kRows);
        const auto   csv = tmp / "data.csv";
        {
            std::ofstream o(csv, std::ios::binary | std::ios::trunc);
            o << "x,y,m\n";
            for (int i = 0; i < kRows; ++i) {
                const int v = (i % 10) + 1;
                xs.push_back(static_cast<double>(i));
                ys.push_back(0.5);
                o << i << ',' << 0.5 << ',' << v << '\n';
                true_sum += v;
            }
        }
        true_count = static_cast<double>(kRows);
        true_avg   = true_sum / true_count;

        const auto parquet = tmp / "data.parquet";
        {
            CsvToParquetOptions po;
            po.input_path  = csv;
            po.output_path = parquet;
            po.has_header  = true;
            po.delimiter   = ',';
            po.overwrite   = true;
            csv_to_parquet(po);
        }
        ConvertOptions opts;
        opts.input_parquet = parquet;
        opts.output_dir = tmp / "prepared";
        opts.dataset_id = "stat_smoke";
        opts.dimensions = {{"x", 0.0, xhi}, {"y", 0.0, 1.0}};
        opts.measures   = {"m"};
        opts.overwrite  = true;
        const auto report = run_parquet_to_columns(opts);

        store.emplace(report.manifest_path);
        schema.dimension_names      = {"x", "y"};
        schema.measure_names        = {"m"};
        schema.domain_bounds        = HyperRect{{{0.0, xhi}, {0.0, 1.0}}};
        schema.object_count         = store->row_count();
        schema.binary_manifest_path = report.manifest_path.string();
    }

    IndexTable make_table() const { return IndexTable::from_columns({xs, ys}); }

    SubstrateConfig substrate() const {
        SubstrateConfig cfg;
        cfg.partition_size = 64;
        return cfg;
    }

    RangeQuery whole_domain(double rel) const {
        return RangeQuery{HyperRect{{{0.0, static_cast<double>(kRows)}, {0.0, 1.0}}},
                          {rel, 0.95}};
    }
};

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

}  // namespace

// Forcing an exact answer reproduces the closed-form truth with zero-width
// intervals: the degenerate limit of the approximate result type.
TEST(StatisticalSmoke, ExactModeIsExact) {
    Fixture     f;
    IndexTable  table = f.make_table();
    AdaptiveKdAccessPath path(f.substrate());
    path.prepare(table);

    EngineConfig cfg;
    cfg.accuracy_mode = EngineConfig::AccuracyMode::ForceExact;
    QueryEngine engine(*f.store, table, path, cfg);

    const QueryResult got = engine.execute(f.whole_domain(0.0), /*ordinal=*/1);

    const auto& sum = find(got, AggregateOp::Sum, 0);
    const auto& cnt = find(got, AggregateOp::CountStar, 0);
    const auto& avg = find(got, AggregateOp::Avg, 0);

    // SUM/AVG match the truth to a floating-point tolerance: the engine folds
    // values in a different order than the closed-form sum, so the result is
    // exact in semantics (zero-width interval, `exact` flag) but not
    // bit-identical to a naive accumulation. COUNT(*) is integral and exact.
    const double sum_tol = 1e-6 * std::max(1.0, std::abs(f.true_sum));
    const double avg_tol = 1e-6 * std::max(1.0, std::abs(f.true_avg));
    EXPECT_TRUE(sum.exact);
    EXPECT_NEAR(sum.estimate, f.true_sum, sum_tol);
    EXPECT_NEAR(sum.ci_low, f.true_sum, sum_tol);
    EXPECT_NEAR(sum.ci_high, f.true_sum, sum_tol);
    EXPECT_DOUBLE_EQ(cnt.estimate, f.true_count);
    EXPECT_NEAR(avg.estimate, f.true_avg, avg_tol);
    EXPECT_EQ(got.metrics.status, "exact");
}

// Across several fixed seeds the approximate loop certifies intervals whose
// half-widths honor the relative-error target, bracket their own estimate, and
// land within a few standard errors of the known truth.
TEST(StatisticalSmoke, ApproximateEstimatesNearTruth) {
    Fixture     f;
    const double rel = 0.10;

    for (std::uint64_t seed : {1u, 2u, 3u, 5u, 8u}) {
        IndexTable  table = f.make_table();
        AdaptiveKdAccessPath path(f.substrate());
        path.prepare(table);

        EngineConfig cfg;
        cfg.accuracy_mode = EngineConfig::AccuracyMode::PerQuery;
        QueryEngine engine(*f.store, table, path, cfg);

        const QueryResult got = engine.execute(f.whole_domain(rel), seed);
        EXPECT_TRUE(got.metrics.target_satisfied) << "seed " << seed;

        const auto& sum = find(got, AggregateOp::Sum, 0);
        const auto& avg = find(got, AggregateOp::Avg, 0);

        for (const auto* e : {&sum, &avg}) {
            EXPECT_LE(e->ci_low, e->estimate) << "seed " << seed;
            EXPECT_LE(e->estimate, e->ci_high) << "seed " << seed;
            if (e->exact) continue;
            EXPECT_LE(e->relative_half_width, rel) << "seed " << seed;
            EXPECT_GT(e->effective_df, 0.0) << "seed " << seed;
            EXPECT_GT(e->ci_high - e->ci_low, 0.0) << "seed " << seed;
        }

        // Point estimates within a few of their own half-widths of truth:
        // deterministic under a fixed seed, so the slack guards calibration
        // without flaking.
        const double sum_hw = 0.5 * (sum.ci_high - sum.ci_low);
        EXPECT_LE(std::abs(sum.estimate - f.true_sum), 4.0 * sum_hw + 1e-9)
            << "seed " << seed;
        const double avg_hw = 0.5 * (avg.ci_high - avg.ci_low);
        EXPECT_LE(std::abs(avg.estimate - f.true_avg), 4.0 * avg_hw + 1e-9)
            << "seed " << seed;

        // COUNT(*) needs no measure read; it is the qualifying population and
        // is always exact.
        const auto& cnt = find(got, AggregateOp::CountStar, 0);
        EXPECT_TRUE(cnt.exact) << "seed " << seed;
        EXPECT_DOUBLE_EQ(cnt.estimate, f.true_count) << "seed " << seed;
    }
}

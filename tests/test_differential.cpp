// Differential test: the engine in exact mode must agree with the full-scan
// oracle over a battery of rectangles, including the degenerate ones (empty,
// full domain, single row, an all-missing measure). Counts agree exactly;
// floating-point aggregates agree to a tight relative tolerance because they
// are reduced through running moments rather than a plain row-order sum.

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

#include "a3i/access_path/substrate_factory.hpp"
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

constexpr double kDomain = 100.0;

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        std::mt19937_64    rng(rd());
        path_ = fs::temp_directory_path() / ("a3i_diff_" + std::to_string(rng()));
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

// A pseudo-random 2D dataset with two measures: m0 always present, m1 missing
// for roughly a third of the rows, so AVG over a region can be NaN. Generated
// deterministically so the oracle comparison is reproducible.
struct Fixture {
    TempDir                          tmp;
    std::optional<BinaryColumnStore> store;
    DatasetSchema                    schema;
    std::vector<double>              xs, ys;

    explicit Fixture(int n = 600) {
        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<double> coord(0.0, kDomain);
        std::uniform_real_distribution<double> val(-50.0, 50.0);
        std::uniform_int_distribution<int>     present(0, 2);

        const auto csv = tmp / "data.csv";
        {
            std::ofstream o(csv, std::ios::binary | std::ios::trunc);
            o << "x,y,m0,m1\n";
            for (int i = 0; i < n; ++i) {
                const double x = coord(rng);
                const double y = coord(rng);
                xs.push_back(x);
                ys.push_back(y);
                o << x << ',' << y << ',' << val(rng) << ',';
                if (present(rng) != 0) o << val(rng);
                o << '\n';
            }
        }
        ConvertOptions opts;
        opts.input_csv  = csv;
        opts.output_dir = tmp / "prepared";
        opts.dataset_id = "diff_fixture";
        opts.has_header = true;
        opts.delimiter  = ',';
        opts.dimensions = {{"x", 0.0, kDomain}, {"y", 0.0, kDomain}};
        opts.measures   = {"m0", "m1"};
        opts.overwrite  = true;
        const auto report = run_csv_to_columns(opts);

        store.emplace(report.manifest_path);
        schema.dimension_names      = {"x", "y"};
        schema.measure_names        = {"m0", "m1"};
        schema.domain_bounds        = HyperRect{{{0.0, kDomain}, {0.0, kDomain}}};
        schema.object_count         = store->row_count();
        schema.binary_manifest_path = report.manifest_path.string();
    }

    IndexTable make_table() const { return IndexTable::from_columns({xs, ys}); }

    SubstrateConfig substrate() const {
        SubstrateConfig cfg;
        cfg.domain_bounds        = HyperRect{{{0.0, kDomain}, {0.0, kDomain}}};
        cfg.refinement_threshold = 32;
        return cfg;
    }
};

RangeQuery rect(double xlo, double xhi, double ylo, double yhi) {
    return RangeQuery{HyperRect{{{xlo, xhi}, {ylo, yhi}}}, {0.0, 0.95}};
}

// Counts are integer-exact; Sum/Avg are reduced through running moments, so a
// fully-read stratum may differ from a plain row-order oracle sum in the last
// bits. Require exact equality on counts and a tight relative tolerance on the
// floating-point aggregates.
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
        } else {
            const double tol = 1e-6 * std::max(1.0, std::abs(o.estimate));
            EXPECT_NEAR(e.estimate, o.estimate, tol) << "aggregate " << i;
        }
    }
}

// Run one rectangle through a freshly-built engine (independent state) over
// every registered substrate and compare each to the oracle. Exact agreement
// must not depend on which substrate partitions the data.
void check(const Fixture& f, const RangeQuery& q) {
    const QueryResult oracle = exact_scan(*f.store, f.schema, q);
    for (const std::string& id : SubstrateFactory::instance().registered_ids()) {
        SCOPED_TRACE(id);
        IndexTable table = f.make_table();
        auto path = SubstrateFactory::instance().create(id, f.substrate());
        path->prepare(table);
        EngineConfig cfg;
        cfg.accuracy_mode = EngineConfig::AccuracyMode::ForceExact;
        QueryEngine engine(*f.store, table, *path, cfg);

        const QueryResult got = engine.execute(q, 1);
        expect_matches_oracle(got, oracle);
    }
}

}  // namespace

TEST(Differential, FullDomain) {
    Fixture f;
    check(f, rect(0.0, kDomain, 0.0, kDomain));
}

TEST(Differential, EmptyRectangle) {
    Fixture f;
    check(f, rect(200.0, 300.0, 200.0, 300.0));
}

TEST(Differential, SubsetRectangles) {
    Fixture f;
    check(f, rect(10.0, 60.0, 20.0, 80.0));
    check(f, rect(0.0, 50.0, 50.0, 100.0));
    check(f, rect(33.0, 66.0, 33.0, 66.0));
    check(f, rect(5.0, 95.0, 40.0, 45.0));
}

TEST(Differential, SingleRow) {
    Fixture f;
    // A pin-prick rectangle around the first generated point.
    const double x = f.xs[0];
    const double y = f.ys[0];
    check(f, rect(x - 0.001, x + 0.001, y - 0.001, y + 0.001));
}

TEST(Differential, RandomBattery) {
    Fixture f;
    std::mt19937_64 rng(999);
    std::uniform_real_distribution<double> lo(0.0, kDomain);
    for (int i = 0; i < 40; ++i) {
        double xlo = lo(rng), xhi = lo(rng);
        double ylo = lo(rng), yhi = lo(rng);
        if (xlo > xhi) std::swap(xlo, xhi);
        if (ylo > yhi) std::swap(ylo, yhi);
        check(f, rect(xlo, xhi, ylo, yhi));
    }
}

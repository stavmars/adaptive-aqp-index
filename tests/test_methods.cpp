// Method presets over the adaptive substrate: the named behaviors compose with
// one engine, exact-mode runs reproduce the oracle, and the single-variable
// persistence behavior holds -- a summary-keeping run reuses a contained
// partition with zero measure reads on a repeat while a no-reuse run re-reads.

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
#include "a3i/engine/method.hpp"
#include "a3i/engine/query_engine.hpp"
#include "a3i/experiments/truth_store.hpp"
#include "a3i/storage/binary_column_store.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/adaptive_kd_access_path.hpp"
#include "a3i/substrates/static_kd_access_path.hpp"
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
        path_ = fs::temp_directory_path() / ("a3i_methods_" + std::to_string(rng()));
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

// A grid dataset with two measures. m0 is always present (x * 100 + y); m1 is
// present only where x < 15, so the far corner has an entirely-missing measure.
struct Fixture {
    TempDir                          tmp;
    std::optional<BinaryColumnStore> store;
    DatasetSchema                    schema;
    std::vector<double>              xs, ys;

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
        opts.dataset_id = "methods_fixture";
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

void expect_agree(const QueryResult& a, const QueryResult& b) {
    ASSERT_EQ(a.aggregates.size(), b.aggregates.size());
    for (std::size_t i = 0; i < a.aggregates.size(); ++i) {
        const auto& x = a.aggregates[i];
        const auto& y = b.aggregates[i];
        if (std::isnan(x.estimate)) {
            EXPECT_TRUE(std::isnan(y.estimate)) << "aggregate " << i;
        } else {
            const double tol = 1e-6 * std::max(1.0, std::abs(x.estimate));
            EXPECT_NEAR(x.estimate, y.estimate, tol) << "aggregate " << i;
        }
    }
}

}  // namespace

// The four presets map to the two-flag behavior identity, and the run id joins
// the substrate with the behavior suffix.
TEST(Methods, PresetsAndNaming) {
    using AM = EngineConfig::AccuracyMode;

    EXPECT_EQ(behavior_config(Behavior::Plain).accuracy_mode, AM::ForceExact);
    EXPECT_FALSE(behavior_config(Behavior::Plain).persist_summaries);

    EXPECT_EQ(behavior_config(Behavior::Agg).accuracy_mode, AM::ForceExact);
    EXPECT_TRUE(behavior_config(Behavior::Agg).persist_summaries);

    EXPECT_EQ(behavior_config(Behavior::Sampling).accuracy_mode, AM::PerQuery);
    EXPECT_FALSE(behavior_config(Behavior::Sampling).persist_summaries);

    EXPECT_EQ(behavior_config(Behavior::A3i).accuracy_mode, AM::PerQuery);
    EXPECT_TRUE(behavior_config(Behavior::A3i).persist_summaries);

    EXPECT_EQ(run_id("adkd", Behavior::Plain), "adkd");
    EXPECT_EQ(run_id("adkd", Behavior::Agg), "adkd_agg");
    EXPECT_EQ(run_id("adkd", Behavior::Sampling), "adkd_sampling");
    EXPECT_EQ(run_id("adkd", Behavior::A3i), "adkd_a3i");
}

// The substrate, not the behavior, decides whether the structure may be cracked
// and whether summaries can be materialized eagerly.
TEST(Methods, ResolvedFlagsAreSubstrateDerived) {
    Fixture f;
    IndexTable t1 = f.make_table();
    IndexTable t2 = f.make_table();
    AdaptiveKdAccessPath adaptive(f.substrate());
    StaticKdAccessPath   stat(f.substrate());
    adaptive.prepare(t1);
    stat.prepare(t2);

    const auto adkd_agg = ResolvedRunConfig::resolve(behavior_config(Behavior::Agg), adaptive);
    EXPECT_TRUE(adkd_agg.allow_refine);
    EXPECT_FALSE(adkd_agg.eager_materialize);  // lazy: the tree is not pre-built

    const auto kd_agg = ResolvedRunConfig::resolve(behavior_config(Behavior::Agg), stat);
    EXPECT_FALSE(kd_agg.allow_refine);
    EXPECT_TRUE(kd_agg.eager_materialize);  // eager: fully-built + summary-keeping

    const auto kd_plain = ResolvedRunConfig::resolve(behavior_config(Behavior::Plain), stat);
    EXPECT_FALSE(kd_plain.allow_refine);
    EXPECT_FALSE(kd_plain.eager_materialize);  // no summaries to keep
}

// plain, agg and a3i all reproduce the oracle exactly over the adaptive
// substrate (counts to the bit, SUM/AVG to a tight tolerance).
TEST(Methods, ExactModeReproducesOracle) {
    Fixture f;
    const RangeQuery exact_q = query(2.0, 12.0, 3.0, 17.0, /*rel=*/0.0);
    const QueryResult oracle = exact_scan(*f.store, f.schema, exact_q);

    for (Behavior b : {Behavior::Plain, Behavior::Agg, Behavior::A3i}) {
        IndexTable table = f.make_table();
        AdaptiveKdAccessPath path(f.substrate());
        path.prepare(table);
        QueryEngine engine(*f.store, table, path, behavior_config(b));
        const QueryResult got = engine.execute(exact_q, /*ordinal=*/1);
        expect_matches_oracle(got, oracle);
    }
}

// Single-variable persistence: a summary-keeping run answers a repeated
// fully-contained query with zero measure reads the second time, while the
// no-reuse run re-reads.
TEST(Methods, AggReusesContainedPartitionAdkdReReads) {
    Fixture f;
    const RangeQuery q = query(0.0, 4.0, 0.0, 4.0, /*rel=*/0.0);

    // adkd_agg: persists the exact summary of the isolated contained partition.
    {
        IndexTable table = f.make_table();
        AdaptiveKdAccessPath path(f.substrate());
        path.prepare(table);
        QueryEngine engine(*f.store, table, path, behavior_config(Behavior::Agg));
        const QueryResult first  = engine.execute(q, 1);
        const QueryResult second = engine.execute(q, 2);
        EXPECT_GT(first.metrics.measure_reads, 0u);
        EXPECT_EQ(second.metrics.measure_reads, 0u);
        const QueryResult oracle = exact_scan(*f.store, f.schema, q);
        expect_matches_oracle(second, oracle);
    }

    // adkd: no persistence, so the second run re-reads.
    {
        IndexTable table = f.make_table();
        AdaptiveKdAccessPath path(f.substrate());
        path.prepare(table);
        QueryEngine engine(*f.store, table, path, behavior_config(Behavior::Plain));
        const QueryResult first  = engine.execute(q, 1);
        const QueryResult second = engine.execute(q, 2);
        EXPECT_GT(first.metrics.measure_reads, 0u);
        EXPECT_GT(second.metrics.measure_reads, 0u);
    }
}

// In exact mode the sampling and full methods reach the same exact answer.
TEST(Methods, SamplingAndA3iAgreeInExactMode) {
    Fixture f;
    const RangeQuery exact_q = query(0.0, 20.0, 0.0, 20.0, /*rel=*/0.0);

    IndexTable t1 = f.make_table();
    AdaptiveKdAccessPath p1(f.substrate());
    p1.prepare(t1);
    QueryEngine sampling(*f.store, t1, p1, behavior_config(Behavior::Sampling));
    const QueryResult got_sampling = sampling.execute(exact_q, 5);

    IndexTable t2 = f.make_table();
    AdaptiveKdAccessPath p2(f.substrate());
    p2.prepare(t2);
    QueryEngine a3i(*f.store, t2, p2, behavior_config(Behavior::A3i));
    const QueryResult got_a3i = a3i.execute(exact_q, 5);

    const QueryResult oracle = exact_scan(*f.store, f.schema, exact_q);
    expect_matches_oracle(got_sampling, oracle);
    expect_matches_oracle(got_a3i, oracle);
    expect_agree(got_sampling, got_a3i);
}

// An approximate a3i run either converges within the target or exactifies, with
// a consistent status and exactification cause.
TEST(Methods, ApproximateRunConvergesOrExactifies) {
    Fixture f;
    IndexTable table = f.make_table();
    AdaptiveKdAccessPath path(f.substrate());
    path.prepare(table);
    QueryEngine engine(*f.store, table, path, behavior_config(Behavior::A3i));

    const double rel = 0.25;
    const RangeQuery q = query(0.0, 20.0, 0.0, 20.0, rel);
    const QueryResult got = engine.execute(q, 9);
    const QueryResult oracle = exact_scan(*f.store, f.schema, q);

    EXPECT_TRUE(got.metrics.target_satisfied);
    const std::string& s = got.metrics.status;
    EXPECT_TRUE(s == "converged" || s == "exactified");
    if (s == "converged") {
        EXPECT_EQ(got.metrics.exactify_cause, "none");
    } else {
        EXPECT_NE(got.metrics.exactify_cause, "none");
    }
    for (std::size_t i = 0; i < got.aggregates.size(); ++i) {
        const auto& e = got.aggregates[i];
        if (e.exact) continue;
        EXPECT_LE(e.relative_half_width, rel) << static_cast<int>(e.op);
        // The estimate must land near the truth, not merely inside its own
        // self-reported interval. The fixed seed keeps this deterministic.
        const auto& o = oracle.aggregates[i];
        if (std::isnan(o.estimate)) continue;
        const double half_width = 0.5 * (e.ci_high - e.ci_low);
        EXPECT_LE(std::abs(e.estimate - o.estimate), 4.0 * half_width)
            << "estimate far from truth for op " << static_cast<int>(e.op);
    }
}

// plain, agg and a3i reproduce the oracle exactly over the fully-built static
// substrate too -- the engine is identical, only the partitioning differs.
TEST(Methods, StaticExactModeReproducesOracle) {
    Fixture f;
    const RangeQuery exact_q = query(2.0, 12.0, 3.0, 17.0, /*rel=*/0.0);
    const QueryResult oracle = exact_scan(*f.store, f.schema, exact_q);

    for (Behavior b : {Behavior::Plain, Behavior::Agg, Behavior::A3i}) {
        IndexTable table = f.make_table();
        StaticKdAccessPath path(f.substrate());
        path.prepare(table);
        QueryEngine engine(*f.store, table, path, behavior_config(b));
        const QueryResult got = engine.execute(exact_q, /*ordinal=*/1);
        expect_matches_oracle(got, oracle);
    }
}

// Eager materialization: over the static substrate the summary-keeping behavior
// pays its measure reads once at initialize(), so even the FIRST fully-contained
// query answers with zero query-time measure reads -- the pay-up-front anchor.
// The plain behavior over the same substrate keeps nothing and re-reads.
TEST(Methods, KdAggEagerZeroQueryReads) {
    Fixture f;
    const RangeQuery q = query(0.0, 20.0, 0.0, 20.0, /*rel=*/0.0);
    const QueryResult oracle = exact_scan(*f.store, f.schema, q);

    {
        IndexTable table = f.make_table();
        StaticKdAccessPath path(f.substrate());
        path.prepare(table);
        QueryEngine engine(*f.store, table, path, behavior_config(Behavior::Agg));
        const QueryResult first = engine.execute(q, 1);
        EXPECT_EQ(first.metrics.measure_reads, 0u);
        expect_matches_oracle(first, oracle);
    }
    {
        IndexTable table = f.make_table();
        StaticKdAccessPath path(f.substrate());
        path.prepare(table);
        QueryEngine engine(*f.store, table, path, behavior_config(Behavior::Plain));
        const QueryResult first = engine.execute(q, 1);
        EXPECT_GT(first.metrics.measure_reads, 0u);
        expect_matches_oracle(first, oracle);
    }
}

// The same behavior agrees over the two substrates: a real two-substrate gate.
TEST(Methods, SubstrateIndependenceAgrees) {
    Fixture f;
    const RangeQuery exact_q = query(1.0, 13.0, 2.0, 16.0, /*rel=*/0.0);

    for (Behavior b : {Behavior::Plain, Behavior::Agg, Behavior::A3i}) {
        IndexTable t1 = f.make_table();
        AdaptiveKdAccessPath p1(f.substrate());
        p1.prepare(t1);
        QueryEngine adaptive(*f.store, t1, p1, behavior_config(b));
        const QueryResult got_adaptive = adaptive.execute(exact_q, 7);

        IndexTable t2 = f.make_table();
        StaticKdAccessPath p2(f.substrate());
        p2.prepare(t2);
        QueryEngine stat(*f.store, t2, p2, behavior_config(b));
        const QueryResult got_static = stat.execute(exact_q, 7);

        expect_agree(got_adaptive, got_static);
    }
}


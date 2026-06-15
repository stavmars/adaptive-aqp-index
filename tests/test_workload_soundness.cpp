// End-to-end engine soundness over a realistic query workload against ONE
// shared, progressively-cracked adaptive substrate.
//
// The per-query coverage test (test_statistical_coverage) deliberately throws
// the index away between trials so its draws are independent. That isolates the
// estimator but never exercises the thing the engine actually does in
// production: answer a long sequence of queries against a single index that
// keeps cracking and keeps accumulating persisted summaries. This test fills
// that gap on a larger fixture (200K rows, the production crack threshold) and
// over a long sequence (500 queries), checking three properties on the SAME
// evolving substrate:
//
//   1. Exact mode reproduces the scan oracle on every query no matter how far
//      the tree has cracked (full statistics, no sampling).
//   2. The approximate loop's certified intervals stay sound on every query
//      (each interval brackets its own estimate, has positive degrees of
//      freedom, meets the requested width, and lands near the truth). The
//      aggregate SUM coverage over the whole run is reported as a sanity
//      diagnostic; the rigorous, independent-trial coverage calibration lives
//      in test_statistical_coverage.
//   3. With persistence on, a region that has been read once is later answered
//      from its persisted summary with zero measure reads, and the reused
//      answer still equals the oracle.
//
// The dataset is generated and ingested through the real CSV->columns path so
// the test also covers the converter and the on-disk column store, not just the
// in-memory query layer. Size is irrelevant to the statistical claims; it is
// chosen so the tree cracks into many partitions and a single query routinely
// decomposes into several strata -- the regime where a grid-then-split index
// would produce tiny under-covering strata, which the gated single-root crack
// here is meant to avoid.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <utility>
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

constexpr std::size_t kRows    = 200000;  // large enough to crack at the default gate
constexpr int         kQueries = 500;     // long sequence for the cumulative checks

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        std::mt19937_64    rng(rd());
        path_ = fs::temp_directory_path() / ("a3i_stat_wl_" + std::to_string(rng()));
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

// An integer row range [lo, hi). Because row i sits at x == i, a rectangle on x
// over [lo, hi) selects exactly rows lo..hi-1, giving a closed-form oracle key.
struct Rect {
    int lo = 0;
    int hi = 0;
    bool operator<(const Rect& o) const {
        return lo != o.lo ? lo < o.lo : hi < o.hi;
    }
};

// The shared 200K-row, single-measure fixture. Built once for the whole suite;
// every test constructs its own substrate/engine over this immutable store so
// the cracking state of one test never leaks into another.
struct Workload {
    TempDir                          tmp;
    std::optional<BinaryColumnStore> store;
    DatasetSchema                    schema;
    std::vector<double>              xs, ys;
    double                           xhi = 0.0;

    explicit Workload(const std::vector<double>& values) {
        const std::size_t n = values.size();
        xhi = static_cast<double>(n);
        const auto csv = tmp / "data.csv";
        {
            std::ofstream o(csv, std::ios::binary | std::ios::trunc);
            o << "x,y,m\n";
            for (std::size_t i = 0; i < n; ++i) {
                xs.push_back(static_cast<double>(i));
                ys.push_back(0.5);
                o << i << ',' << 0.5 << ',' << values[i] << '\n';
            }
        }
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
        opts.dataset_id = "stat_workload";
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

    // The production substrate configuration: a single root with the default
    // crack gate (partition_size == 1024), deliberately not overridden.
    SubstrateConfig substrate() const {
        SubstrateConfig cfg;
        return cfg;
    }

    RangeQuery query(const Rect& r, double rel) const {
        return RangeQuery{HyperRect{{{static_cast<double>(r.lo),
                                      static_cast<double>(r.hi)},
                                     {0.0, 1.0}}},
                          {rel, 0.95}};
    }
};

// Static fixture: the heavy 200K ingest happens once for the whole suite.
class WorkloadSoundness : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        std::vector<double>              values(kRows);
        std::mt19937_64                  gen(2024);
        std::normal_distribution<double> normal(50.0, 15.0);  // well-behaved, CV = 0.3
        for (std::size_t i = 0; i < kRows; ++i) values[i] = normal(gen);
        wl_      = new Workload(values);
        queries_ = build_queries();
    }

    static void TearDownTestSuite() {
        delete wl_;
        wl_ = nullptr;
        oracle_cache_.clear();
        queries_.clear();
    }

    // A deterministic mix of repeated "hot" rectangles (which become contained
    // partitions once the tree cracks, exercising reuse) and fresh random
    // rectangles of varying width (which keep driving new cracks). Every width
    // exceeds the crack threshold so the tree actually subdivides.
    static std::vector<Rect> build_queries() {
        const int      n = static_cast<int>(kRows);
        std::mt19937_64 gen(7);

        std::vector<Rect> hot;
        for (int k = 0; k < 8; ++k) {
            const int width = 4000 + 2500 * k;            // 4000 .. 21500
            const int lo    = (n / 10) * k % (n - width);  // spread across the domain
            hot.push_back({lo, lo + width});
        }

        std::uniform_real_distribution<double> coin(0.0, 1.0);
        std::uniform_int_distribution<int>     hot_pick(0, static_cast<int>(hot.size()) - 1);
        std::uniform_int_distribution<int>     width_pick(1500, 40000);

        std::vector<Rect> qs;
        qs.reserve(kQueries);
        for (int i = 0; i < kQueries; ++i) {
            if (coin(gen) < 0.35) {
                qs.push_back(hot[hot_pick(gen)]);  // repeat -> contained / reuse
            } else {
                const int width = width_pick(gen);
                std::uniform_int_distribution<int> lo_pick(0, n - width);
                const int lo = lo_pick(gen);
                qs.push_back({lo, lo + width});
            }
        }
        return qs;
    }

    // Lazily-computed, cached scan-oracle answer for a rectangle. Shared across
    // all tests because the underlying store is immutable.
    static const QueryResult& oracle_for(const Rect& r) {
        auto it = oracle_cache_.find(r);
        if (it != oracle_cache_.end()) return it->second;
        const QueryResult res =
            exact_scan(*wl_->store, wl_->schema, wl_->query(r, /*rel=*/0.0));
        return oracle_cache_.emplace(r, res).first->second;
    }

    static Workload*                 wl_;
    static std::map<Rect, QueryResult> oracle_cache_;
    static std::vector<Rect>         queries_;
};

Workload*                  WorkloadSoundness::wl_ = nullptr;
std::map<Rect, QueryResult> WorkloadSoundness::oracle_cache_;
std::vector<Rect>          WorkloadSoundness::queries_;

// The lowest empirical coverage we accept before flagging a calibration
// regression. Under well-calibrated 95% intervals the count of covered answers
// over n trials is approximately Binomial(n, 0.95); we fail only when observed
// coverage falls more than `z` binomial standard errors below nominal. The
// floor therefore tightens automatically as n grows -- no hand-tuned constant.
double coverage_floor(int n, double z) {
    if (n <= 0) return 0.0;
    constexpr double kNominal = 0.95;
    const double se = std::sqrt(kNominal * (1.0 - kNominal) / n);
    return std::max(0.0, kNominal - z * se);
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

// Every aggregate of an exact answer must equal the oracle: counts to the bit,
// SUM/AVG to a tight relative tolerance (the engine folds via Welford, so it
// need not match the oracle's row-order sum to the last ULP).
void expect_exact_matches_oracle(const QueryResult& got, const QueryResult& oracle,
                                 int qi) {
    ASSERT_EQ(got.aggregates.size(), oracle.aggregates.size()) << "query " << qi;
    for (const auto& e : got.aggregates) {
        EXPECT_TRUE(e.exact) << "query " << qi << " op " << static_cast<int>(e.op);
        const auto& o = find(oracle, e.op, e.measure_id);
        const bool is_count = e.op == AggregateOp::CountMeasure ||
                              e.op == AggregateOp::CountStar;
        if (std::isnan(o.estimate)) {
            EXPECT_TRUE(std::isnan(e.estimate)) << "query " << qi;
        } else if (is_count) {
            EXPECT_DOUBLE_EQ(e.estimate, o.estimate) << "query " << qi;
        } else {
            const double tol = 1e-6 * std::max(1.0, std::abs(o.estimate));
            EXPECT_NEAR(e.estimate, o.estimate, tol) << "query " << qi;
        }
    }
}

}  // namespace

// Exact mode over the long workload: a single substrate cracks further on every
// query, yet every aggregate still equals the scan oracle. This is the
// adaptive-substrate exact-correctness gate under cumulative cracking -- full
// statistics, no sampling.
TEST_F(WorkloadSoundness, ExactModeReproducesOracleUnderCumulativeCracking) {
    IndexTable           table = wl_->make_table();
    AdaptiveKdAccessPath path(wl_->substrate());
    path.prepare(table);

    EngineConfig cfg;
    cfg.accuracy_mode     = EngineConfig::AccuracyMode::ForceExact;
    cfg.persist_summaries = false;
    QueryEngine engine(*wl_->store, table, path, cfg);

    std::uint64_t total_refined = 0;
    for (int i = 0; i < kQueries; ++i) {
        const Rect&       r  = queries_[i];
        const QueryResult got =
            engine.execute(wl_->query(r, /*rel=*/0.05),
                           static_cast<std::uint64_t>(i + 1));
        EXPECT_EQ(got.metrics.status, "exact") << "query " << i;
        EXPECT_EQ(got.metrics.exactify_cause, "none") << "query " << i;
        expect_exact_matches_oracle(got, oracle_for(r), i);
        total_refined += got.metrics.partitions_refined;
    }

    // The workload must actually have exercised cracking, otherwise the
    // "under cumulative cracking" claim is vacuous.
    EXPECT_GT(total_refined, 0u) << "the tree never cracked";
}

// Approximate mode over the long workload: certified intervals stay sound on
// every query (bracket their own estimate, positive df, within target width)
// and land near the truth. The aggregate SUM coverage over the whole evolving
// run is reported as a sanity diagnostic against a binomial floor; the rigorous
// independent-trial calibration claim lives in test_statistical_coverage.
TEST_F(WorkloadSoundness, ApproximateIntervalsStaySoundAndCalibrated) {
    IndexTable           table = wl_->make_table();
    AdaptiveKdAccessPath path(wl_->substrate());
    path.prepare(table);

    EngineConfig cfg;
    cfg.accuracy_mode     = EngineConfig::AccuracyMode::PerQuery;
    cfg.persist_summaries = false;
    // Interval calibration is a property of the sampling math, independent of
    // storage. Run it on an in-memory store: the fixture's rows are contiguous,
    // so a range query's rows are dense over a narrow span and the on-disk cost
    // model would (correctly) read them whole rather than sample. Eager has no
    // scan path, so the loop genuinely samples here; on-disk scan-to-exactify
    // is covered by the dedicated on-disk tests.
    BinaryColumnStore eager(wl_->schema.binary_manifest_path, /*selected=*/{},
                            MeasureStorage::Eager);
    QueryEngine engine(eager, table, path, cfg);

    const double rel     = 0.10;
    int          sampled = 0;
    int          covered = 0;

    for (int i = 0; i < kQueries; ++i) {
        const Rect&       r      = queries_[i];
        const QueryResult oracle = oracle_for(r);
        const QueryResult got =
            engine.execute(wl_->query(r, rel), static_cast<std::uint64_t>(i + 1));

        EXPECT_TRUE(got.metrics.target_satisfied) << "query " << i;

        // COUNT(*) is structural and must always be exact.
        const auto& cs = find(got, AggregateOp::CountStar, 0);
        EXPECT_TRUE(cs.exact) << "query " << i;
        EXPECT_DOUBLE_EQ(cs.estimate, find(oracle, AggregateOp::CountStar, 0).estimate)
            << "query " << i;

        // SUM is the tracked aggregate for the coverage statistic.
        const auto&  e        = find(got, AggregateOp::Sum, 0);
        const auto&  o        = find(oracle, AggregateOp::Sum, 0);
        const double true_sum = o.estimate;
        const double tol      = 1e-6 * std::max(1.0, std::abs(true_sum));

        if (e.exact) {
            EXPECT_NEAR(e.estimate, true_sum, tol) << "query " << i;
            continue;
        }

        // Interval soundness on every approximate answer.
        EXPECT_LE(e.relative_half_width, rel + 1e-9) << "query " << i;
        EXPECT_LE(e.ci_low, e.estimate) << "query " << i;
        EXPECT_LE(e.estimate, e.ci_high) << "query " << i;
        EXPECT_GT(e.effective_df, 0.0) << "query " << i;

        // Self-consistency-vs-accuracy guard: the estimate must land near the
        // truth, not merely inside a self-reported interval.
        const double half_width = 0.5 * (e.ci_high - e.ci_low);
        EXPECT_LE(std::abs(e.estimate - true_sum), 5.0 * half_width)
            << "estimate far from truth, query " << i;

        ++sampled;
        if (e.ci_low - tol <= true_sum && true_sum <= e.ci_high + tol) ++covered;
    }

    const double coverage =
        sampled == 0 ? 0.0 : static_cast<double>(covered) / sampled;
    // Two binomial floors: a strict one (3 SE) is the independent-trial
    // reference, and a relaxed one (6 SE) is the actual gate here. The trials
    // share one evolving tree, so they are positively correlated and the
    // binomial model understates the true sampling spread; widening the band is
    // the principled response to that correlation, and it keeps this end-to-end
    // test a soundness check rather than a second calibration gate.
    const double strict_floor  = coverage_floor(sampled, /*z=*/3.0);
    const double relaxed_floor = coverage_floor(sampled, /*z=*/6.0);
    RecordProperty("sampled", sampled);
    RecordProperty("coverage_x1000", static_cast<int>(std::lround(coverage * 1000)));
    RecordProperty("strict_floor_x1000",
                   static_cast<int>(std::lround(strict_floor * 1000)));
    std::cout << "[ workload ] sampled=" << sampled << '/' << kQueries
              << " SUM coverage=" << coverage << " (nominal 0.95; strict 3-SE floor="
              << strict_floor << ", gate 6-SE floor=" << relaxed_floor << ")"
              << std::endl;

    // Enough approximate answers to make the coverage statistic meaningful.
    EXPECT_GE(sampled, 50) << "too few approximate answers to judge coverage";
    EXPECT_GE(coverage, relaxed_floor) << "workload SUM coverage too low";
}

// Persistence over the long workload: every answer stays exact, and a region
// that has already been read in full is later answered from its persisted
// summary with zero measure reads.
TEST_F(WorkloadSoundness, PersistentSummariesReuseContainedRegionsAndStayExact) {
    EngineConfig cfg;
    cfg.accuracy_mode     = EngineConfig::AccuracyMode::ForceExact;
    cfg.persist_summaries = true;

    // Part A -- the read->reuse transition on a fresh substrate. The first
    // execution reads the region in full (cracking isolates it and persistence
    // stores its exact summary); the identical repeat is answered from that
    // summary with no measure reads, and still equals the oracle.
    {
        IndexTable           table = wl_->make_table();
        AdaptiveKdAccessPath path(wl_->substrate());
        path.prepare(table);
        QueryEngine engine(*wl_->store, table, path, cfg);

        const Rect        hot{queries_.front().lo, queries_.front().hi};
        const QueryResult first  = engine.execute(wl_->query(hot, 0.0), 9001);
        const QueryResult second = engine.execute(wl_->query(hot, 0.0), 9002);
        EXPECT_GT(first.metrics.measure_reads, 0u) << "first read of the region read nothing";
        EXPECT_EQ(second.metrics.measure_reads, 0u) << "contained repeat still read measures";
        expect_exact_matches_oracle(first, oracle_for(hot), -1);
        expect_exact_matches_oracle(second, oracle_for(hot), -2);
    }

    // Part B -- correctness and reuse over the full workload on its own
    // substrate. Every answer stays exact no matter how persisted state and
    // cracking interact, and repeats that land on an already-complete contained
    // region are answered with zero measure reads, so reuse genuinely happens
    // during a realistic run, not just in the isolated Part-A pair.
    {
        IndexTable           table = wl_->make_table();
        AdaptiveKdAccessPath path(wl_->substrate());
        path.prepare(table);
        QueryEngine engine(*wl_->store, table, path, cfg);

        int zero_read_answers = 0;
        for (int i = 0; i < kQueries; ++i) {
            const Rect&       r  = queries_[i];
            const QueryResult got =
                engine.execute(wl_->query(r, /*rel=*/0.0),
                               static_cast<std::uint64_t>(i + 1));
            expect_exact_matches_oracle(got, oracle_for(r), i);
            if (got.metrics.measure_reads == 0) ++zero_read_answers;
        }
        RecordProperty("zero_read_answers", zero_read_answers);
        std::cout << "[ workload ] zero-read (reused) answers=" << zero_read_answers
                  << '/' << kQueries << std::endl;
        EXPECT_GT(zero_read_answers, 0) << "persistence never reused a region in the run";
    }
}

// Empirical confidence-interval coverage: the honest calibration check.
//
// Over many seeds and several rectangles on fixtures with KNOWN aggregates we
// measure how often the certified 95% interval actually brackets the true SUM.
// For a well-behaved measure the empirical coverage of the sampled intervals
// must sit near the nominal 0.95. For a deliberately heavy-tailed measure --
// almost all tiny values with rare enormous ones -- early samples routinely
// miss the tail, the observed variance collapses, the loop converges on a too
// tight interval, and coverage falls below nominal.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "a3i/core/query.hpp"
#include "a3i/core/schema.hpp"
#include "a3i/engine/query_engine.hpp"
#include "a3i/experiments/truth_store.hpp"
#include "a3i/storage/binary_column_store.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/adaptive_kd_access_path.hpp"
#include "a3i/substrates/grid_akd_access_path.hpp"
#include "a3i/tools/parquet_to_columns_pipeline.hpp"
#include "a3i/tools/csv_to_parquet.hpp"

namespace fs = std::filesystem;

namespace {

using namespace a3i;

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        std::mt19937_64    rng(rd());
        path_ = fs::temp_directory_path() / ("a3i_stat_cov_" + std::to_string(rng()));
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

// A single-measure dataset built from an explicit value column. Row i sits at
// (i, 0.5) so every row has a distinct x coordinate and rectangles on x select
// a contiguous, known prefix/suffix of the rows. An optional null mask marks
// rows whose measure value is missing: those are written as an empty CSV field,
// which the converter encodes as NaN. COUNT(measure) then counts only the
// non-null rows and is genuinely distinct from the exact COUNT(*).
struct Dataset {
    TempDir                          tmp;
    std::optional<BinaryColumnStore> store;
    DatasetSchema                    schema;
    std::vector<double>              xs, ys;
    double                           xhi = 0.0;

    Dataset(const std::string& id, const std::vector<double>& values,
            const std::vector<char>* nulls = nullptr) {
        const std::size_t n = values.size();
        xhi = static_cast<double>(n);
        const auto csv = tmp / "data.csv";
        {
            std::ofstream o(csv, std::ios::binary | std::ios::trunc);
            o << "x,y,m\n";
            for (std::size_t i = 0; i < n; ++i) {
                xs.push_back(static_cast<double>(i));
                ys.push_back(0.5);
                o << i << ',' << 0.5 << ',';
                if (nulls != nullptr && (*nulls)[i]) {
                    o << '\n';  // empty measure field -> null -> NaN
                } else {
                    o << values[i] << '\n';
                }
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
        opts.dataset_id = id;
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

    RangeQuery query(double xlo, double xhi_q, double rel) const {
        return RangeQuery{HyperRect{{{xlo, xhi_q}, {0.0, 1.0}}}, {rel, 0.95}};
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

struct CoverageStats {
    int trials = 0;
    int covered = 0;
    int sampled = 0;          // trials answered approximately (not exactified)
    int sampled_covered = 0;

    double overall() const {
        return trials == 0 ? 0.0 : static_cast<double>(covered) / trials;
    }
    double sampled_coverage() const {
        return sampled == 0 ? 0.0 : static_cast<double>(sampled_covered) / sampled;
    }
};

// The lowest empirical coverage we accept before flagging a calibration
// regression. Under well-calibrated 95% intervals the count of covered answers
// over n independent trials is approximately Binomial(n, 0.95); we fail only
// when observed coverage falls more than `z` binomial standard errors below
// nominal. The floor tightens automatically as n grows, so there is no
// hand-tuned acceptance constant.
double coverage_floor(int n, double z = 3.0) {
    if (n <= 0) return 0.0;
    constexpr double kNominal = 0.95;
    const double se = std::sqrt(kNominal * (1.0 - kNominal) / n);
    return std::max(0.0, kNominal - z * se);
}

// Sweep several x-rectangles, each over many seeds, accumulating how often the
// certified interval for `op` brackets the rectangle's true aggregate value.
// Each (rectangle, seed) trial uses a fresh table/path/engine so the draws are
// independent and no cracking carries over.
CoverageStats sweep(const Dataset& d, double rel, int seeds,
                    AggregateOp op = AggregateOp::Sum,
                    double outlier_budget = 0.0) {
    // Coverage is a property of the sampling math, independent of storage. Run
    // it on an in-memory store: these fixtures are tiny / contiguous, so on disk
    // the access-path cost model would (correctly) read them whole rather than
    // sample, leaving no sampled intervals to score. Eager has no scan path, so
    // the loop genuinely samples here; on-disk scan-to-exactify is covered by
    // the dedicated on-disk tests.
    BinaryColumnStore eager(d.schema.binary_manifest_path, /*selected=*/{},
                            MeasureStorage::Eager);
    const double            n = d.xhi;
    const std::vector<std::pair<double, double>> rects = {
        {0.0, n},
        {0.0, 0.5 * n},
        {0.25 * n, n},
        {0.1 * n, 0.9 * n},
        {0.0, 0.75 * n},
        {0.25 * n, 0.75 * n},
    };

    CoverageStats cs;
    for (const auto& [xlo, xhi_q] : rects) {
        const RangeQuery oq = d.query(xlo, xhi_q, /*rel=*/0.0);
        const QueryResult oracle = exact_scan(*d.store, d.schema, oq);
        const double true_val = find(oracle, op, 0).estimate;
        const double tol = 1e-6 * std::max(1.0, std::abs(true_val));

        for (int s = 1; s <= seeds; ++s) {
            IndexTable           table = d.make_table();
            AdaptiveKdAccessPath path(d.substrate());
            path.prepare(table);
            EngineConfig cfg;
            cfg.accuracy_mode = EngineConfig::AccuracyMode::PerQuery;
            cfg.outlier_budget_fraction = outlier_budget;
            QueryEngine engine(eager, table, path, cfg);

            const QueryResult got =
                engine.execute(d.query(xlo, xhi_q, rel), static_cast<std::uint64_t>(s));
            const auto& e = find(got, op, 0);
            const bool covered =
                e.ci_low - tol <= true_val && true_val <= e.ci_high + tol;

            ++cs.trials;
            cs.covered += covered ? 1 : 0;
            if (!e.exact) {
                ++cs.sampled;
                cs.sampled_covered += covered ? 1 : 0;
            }
        }
    }
    return cs;
}

}  // namespace

// A well-behaved measure: independent Normal draws with moderate spread. The
// per-rectangle distribution is light-tailed and roughly symmetric, so at the
// sample sizes the loop chooses the Student-t intervals cover near the nominal
// 0.95.
TEST(StatisticalCoverage, WellBehavedMeasureCoversNearNominal) {
    const std::size_t   kN = 20000;
    std::vector<double> values(kN);
    std::mt19937_64                  gen(12345);
    std::normal_distribution<double> normal(50.0, 15.0);  // CV = 0.3
    for (std::size_t i = 0; i < kN; ++i) values[i] = normal(gen);
    Dataset d("cov_well_behaved", values);

    const CoverageStats cs = sweep(d, /*rel=*/0.05, /*seeds=*/80);

    RecordProperty("trials", cs.trials);
    RecordProperty("sampled_trials", cs.sampled);
    RecordProperty("overall_coverage_x1000",
                   static_cast<int>(std::lround(cs.overall() * 1000)));
    RecordProperty("sampled_coverage_x1000",
                   static_cast<int>(std::lround(cs.sampled_coverage() * 1000)));

    std::cout << "[ well-behaved ] sampled trials=" << cs.sampled << '/' << cs.trials
              << " sampled coverage=" << cs.sampled_coverage()
              << " overall coverage=" << cs.overall() << " (nominal 0.95)"
              << std::endl;

    // The sweep must produce a meaningful number of sampled (non-exactified)
    // intervals, otherwise the coverage statistic is vacuous. Thirty is the
    // smallest count at which a binomial coverage decision is worth making.
    EXPECT_GE(cs.sampled, 30) << "too few approximate answers to judge coverage";
    // Empirical coverage at or above the binomial floor (three standard errors
    // below the nominal 0.95 for this many sampled trials): well-calibrated
    // intervals clear it, a genuine calibration regression does not.
    const double floor = coverage_floor(cs.sampled);
    EXPECT_GE(cs.sampled_coverage(), floor)
        << "sampled coverage " << cs.sampled_coverage() << " below binomial floor "
        << floor;
    EXPECT_LE(cs.sampled_coverage(), 1.0);
    EXPECT_GE(cs.overall(), floor) << "overall coverage too low";
}

// A heavy-tailed measure: independent lognormal draws (strong right skew). At
// the modest sample sizes a loose target permits, the symmetric Student-t
// interval is a poor fit for the skewed sampling distribution and coverage
// drifts below nominal. The observed sample variance is used plain -- there is
// no prior-variance floor -- so this under-coverage is a documented
// limitation. We RECORD the measured coverage and never fail on its value;
// the only assertion is that the sweep actually reached the sampling path.
TEST(StatisticalCoverage, HeavyTailUnderCoverageIsRecorded) {
    const std::size_t   kN = 20000;
    std::vector<double> values(kN);
    std::mt19937_64                     gen(99991);
    std::lognormal_distribution<double> heavy(2.0, 1.5);  // strong right skew
    for (std::size_t i = 0; i < kN; ++i) values[i] = heavy(gen);
    Dataset d("cov_heavy_tail", values);

    const CoverageStats cs = sweep(d, /*rel=*/0.30, /*seeds=*/50);

    RecordProperty("trials", cs.trials);
    RecordProperty("sampled_trials", cs.sampled);
    RecordProperty("overall_coverage_x1000",
                   static_cast<int>(std::lround(cs.overall() * 1000)));
    RecordProperty("sampled_coverage_x1000",
                   static_cast<int>(std::lround(cs.sampled_coverage() * 1000)));

    std::cout << "[ heavy-tail ] sampled trials=" << cs.sampled << '/' << cs.trials
              << " sampled coverage=" << cs.sampled_coverage()
              << " overall coverage=" << cs.overall()
              << " (nominal 0.95; under-coverage expected and recorded, not failed)"
              << std::endl;

    // The recorded coverage only means something if the loop actually sampled;
    // the coverage value itself is intentionally not asserted.
    EXPECT_GT(cs.sampled, 0) << "heavy-tail sweep never sampled";
}

// COUNT(measure) coverage on a measure with a genuine null fraction. When a
// measure is never null, COUNT(measure) equals the always-exact COUNT(*) and
// its interval is degenerate, so it never exercises the Bernoulli presence
// estimator. Here roughly 40% of rows are null, so COUNT(measure) is a real
// sampled aggregate -- the estimator scales an observed presence rate up to the
// rectangle and reports a Bernoulli interval. The presence indicator is a
// bounded [0,1] variable, so its sampling distribution is well-behaved and the
// certified 95% interval must cover the true non-null count near nominal.
TEST(StatisticalCoverage, NullBearingCountCoversNearNominal) {
    const std::size_t   kN = 20000;
    std::vector<double> values(kN, 1.0);
    std::vector<char>   nulls(kN, 0);
    std::mt19937_64                        gen(424242);
    std::bernoulli_distribution            present(0.6);  // ~40% null
    for (std::size_t i = 0; i < kN; ++i) nulls[i] = present(gen) ? 0 : 1;
    Dataset d("cov_null_count", values, &nulls);

    const CoverageStats cs =
        sweep(d, /*rel=*/0.05, /*seeds=*/80, AggregateOp::CountMeasure);

    RecordProperty("trials", cs.trials);
    RecordProperty("sampled_trials", cs.sampled);
    RecordProperty("overall_coverage_x1000",
                   static_cast<int>(std::lround(cs.overall() * 1000)));
    RecordProperty("sampled_coverage_x1000",
                   static_cast<int>(std::lround(cs.sampled_coverage() * 1000)));

    std::cout << "[ null-count ] sampled trials=" << cs.sampled << '/' << cs.trials
              << " sampled coverage=" << cs.sampled_coverage()
              << " overall coverage=" << cs.overall() << " (nominal 0.95)"
              << std::endl;

    // The sweep must produce a meaningful number of sampled (non-exact) COUNT
    // intervals, otherwise the coverage statistic is vacuous.
    EXPECT_GE(cs.sampled, 30) << "too few approximate COUNT answers to judge coverage";
    const double floor = coverage_floor(cs.sampled);
    EXPECT_GE(cs.sampled_coverage(), floor)
        << "sampled COUNT coverage " << cs.sampled_coverage()
        << " below binomial floor " << floor;
    EXPECT_LE(cs.sampled_coverage(), 1.0);
    EXPECT_GE(cs.overall(), floor) << "overall COUNT coverage too low";
}

// A near-point-mass tail like the real failure case: almost all values are small
// and a sparse set is enormous, so a sample that misses the tail produces a
// too-low, too-tight interval and SUM coverage collapses. Holding the budget-many
// most-deviant rows out of the sample and contributing them exactly restores the
// interval to a body it can actually cover. We assert both halves: without the
// index coverage is poor, and with it coverage clears the binomial floor and
// improves by a wide margin. (The held-out rows make the answer no less correct:
// the separate exact-mode soundness tests confirm equality with the oracle.)
TEST(StatisticalCoverage, OutlierIndexRestoresHeavyTailCoverage) {
    const std::size_t   kN = 20000;
    std::vector<double> values(kN);
    std::mt19937_64                      gen(20260623);
    std::uniform_real_distribution<double> body(1.0, 10.0);
    std::bernoulli_distribution            spike(0.01);  // ~1% enormous rows
    for (std::size_t i = 0; i < kN; ++i) {
        values[i] = spike(gen) ? 1.0e6 : body(gen);
    }
    Dataset d("cov_point_mass_tail", values);

    // Budget ~ the spike fraction, so the deviation index captures the tail.
    const CoverageStats off = sweep(d, /*rel=*/0.10, /*seeds=*/50,
                                    AggregateOp::Sum, /*outlier_budget=*/0.0);
    const CoverageStats on  = sweep(d, /*rel=*/0.10, /*seeds=*/50,
                                    AggregateOp::Sum, /*outlier_budget=*/0.012);

    RecordProperty("off_sampled_coverage_x1000",
                   static_cast<int>(std::lround(off.sampled_coverage() * 1000)));
    RecordProperty("on_sampled_coverage_x1000",
                   static_cast<int>(std::lround(on.sampled_coverage() * 1000)));
    std::cout << "[ point-mass tail ] SUM sampled coverage: index OFF="
              << off.sampled_coverage() << " (n=" << off.sampled << ")  ON="
              << on.sampled_coverage() << " (n=" << on.sampled << ")  (nominal 0.95)"
              << std::endl;

    ASSERT_GE(on.sampled, 30) << "index-on sweep produced too few sampled intervals";
    // With the index, coverage clears the binomial floor (near nominal)...
    EXPECT_GE(on.sampled_coverage(), coverage_floor(on.sampled))
        << "index-on coverage " << on.sampled_coverage() << " below nominal floor";
    // ...and is a clear improvement over the uncorrected sweep.
    if (off.sampled > 0) {
        EXPECT_GT(on.sampled_coverage(), off.sampled_coverage() + 0.05)
            << "the index did not materially lift coverage (off="
            << off.sampled_coverage() << " on=" << on.sampled_coverage() << ")";
    }
}

// The held-out rows are read once and banked in the partition's summary, then a
// later query over the same (persistent) engine reuses them exactly and reads
// nothing. Run under ForceExact so each contained partition becomes complete on
// the first query: the answer must equal the oracle BOTH times (held-out rows
// counted exactly once -- no double count, no dropped tail), and the second
// query must re-read nothing.
TEST(StatisticalCoverage, OutlierBankingReusesExactlyWithoutReReading) {
    const std::size_t   kN = 4000;
    std::vector<double> values(kN);
    std::mt19937_64                        gen(424242);
    std::uniform_real_distribution<double> body(1.0, 10.0);
    std::bernoulli_distribution            spike(0.01);
    for (std::size_t i = 0; i < kN; ++i) {
        values[i] = spike(gen) ? 1.0e6 : body(gen);
    }
    Dataset d("cov_banking_reuse", values);

    BinaryColumnStore eager(d.schema.binary_manifest_path, /*selected=*/{},
                            MeasureStorage::Eager);
    IndexTable           table = d.make_table();
    AdaptiveKdAccessPath path(d.substrate());
    path.prepare(table);
    EngineConfig cfg;
    cfg.accuracy_mode = EngineConfig::AccuracyMode::ForceExact;  // partitions complete
    cfg.persist_summaries = true;                                // cross-query reuse
    cfg.outlier_budget_fraction = 0.02;
    QueryEngine engine(eager, table, path, cfg);

    const RangeQuery q = d.query(0.0, d.xhi, /*rel=*/0.0);
    const QueryResult oracle = exact_scan(*d.store, d.schema, q);
    const double truth = find(oracle, AggregateOp::Sum, 0).estimate;
    const double tol = 1e-6 * std::max(1.0, std::abs(truth));

    const QueryResult q1 = engine.execute(q, /*ordinal=*/1);
    const auto& s1 = find(q1, AggregateOp::Sum, 0);
    EXPECT_TRUE(s1.exact);
    EXPECT_NEAR(s1.estimate, truth, tol)
        << "first query SUM mis-counts the held-out rows";
    EXPECT_GT(q1.metrics.measure_reads, 0u);

    const QueryResult q2 = engine.execute(q, /*ordinal=*/2);
    const auto& s2 = find(q2, AggregateOp::Sum, 0);
    EXPECT_TRUE(s2.exact);
    EXPECT_NEAR(s2.estimate, truth, tol)
        << "reused SUM mis-counts the held-out rows (double/under count on reuse)";
    EXPECT_EQ(q2.metrics.measure_reads, 0u)
        << "second query re-read instead of reusing the banked summaries";
}

// On disk the held-out rows are appended to the body batch (one gather, not a
// separate round). The appended ids are not globally ascending, so the gather
// must sort them; were it told the batch was already sorted, the sequential-
// scan path would mis-read values. Force the scan path (a tiny under-one-page
// dataset) under ForceExact with the index on, and assert the answer still
// matches the oracle.
TEST(StatisticalCoverage, OutlierBankingOnDiskScanPathReadsCorrectly) {
    const std::size_t   kN = 400;
    std::vector<double> values(kN);
    std::mt19937_64                        gen(99887766);
    std::uniform_real_distribution<double> body(1.0, 10.0);
    std::bernoulli_distribution            spike(0.03);
    for (std::size_t i = 0; i < kN; ++i) {
        values[i] = spike(gen) ? 1.0e6 : body(gen);
    }
    Dataset d("cov_banking_ondisk", values);

    BinaryColumnStore ondisk(d.schema.binary_manifest_path, /*selected=*/{},
                             MeasureStorage::OnDisk);
    IndexTable           table = d.make_table();
    AdaptiveKdAccessPath path(d.substrate());
    path.prepare(table);
    EngineConfig cfg;
    cfg.accuracy_mode = EngineConfig::AccuracyMode::ForceExact;
    cfg.persist_summaries = true;
    cfg.outlier_budget_fraction = 0.03;
    QueryEngine engine(ondisk, table, path, cfg);

    const RangeQuery q = d.query(0.0, d.xhi, /*rel=*/0.0);
    const QueryResult oracle = exact_scan(*d.store, d.schema, q);
    const double truth = find(oracle, AggregateOp::Sum, 0).estimate;
    const double tol = 1e-6 * std::max(1.0, std::abs(truth));

    const QueryResult got = engine.execute(q, /*ordinal=*/1);
    const auto& s = find(got, AggregateOp::Sum, 0);
    EXPECT_TRUE(s.exact);
    EXPECT_NEAR(s.estimate, truth, tol)
        << "on-disk banking mis-read the appended held-out rows";
    EXPECT_GT(got.metrics.scan_path_rows, 0u)
        << "expected the scan path to exercise the appended-outlier read";
    EXPECT_GT(got.metrics.outlier_rows, 0u)
        << "the held-out tail should have been read and banked";
    // Read-work identity: every measure value read is charged to exactly one of
    // sampled / exactified / outlier. Single-measure fixture, so the per-measure
    // multiplier is 1.
    EXPECT_EQ(got.metrics.measure_reads,
              got.metrics.sampled_rows + got.metrics.exactified_rows +
                  got.metrics.outlier_rows)
        << "read-work identity broken (measure_reads != sampled+exactified+outlier)";
}

// The outlier flag column is position-keyed: swap_positions must move a row's
// flag bit in lockstep with the row, so the SET of flagged row-ids stays
// invariant under the in-place swaps that cracking performs. The end-to-end
// banking tests crack swap-free on their x-sorted fixtures, so this exercises
// the carry directly -- the one disjointness invariant otherwise protected only
// by a comment. A broken carry would leave the bit at a stale position (now
// holding a different row), changing the flagged row-id set.
TEST(StatisticalCoverage, OutlierFlagSurvivesSwapPositions) {
    const std::size_t n = 200;
    std::vector<double> xs(n), ys(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = static_cast<double>(i);
        ys[i] = 0.5;
    }
    IndexTable table = IndexTable::from_columns({xs, ys});

    const std::vector<RowId> flagged = {3, 17, 18, 64, 65, 130, 199};
    table.set_flags_by_rowid(flagged);
    const std::set<RowId> expected(flagged.begin(), flagged.end());

    const auto flagged_rowids = [&]() {
        std::set<RowId> s;
        table.for_each_flagged_in_range(
            0, static_cast<IndexPos>(table.size()),
            [&](IndexPos pos) { s.insert(table.row_id(pos)); });
        return s;
    };
    ASSERT_EQ(flagged_rowids(), expected) << "install flagged the wrong rows";

    // Repeatedly shuffle the table in place (what cracking's
    // partition step does), checking the flagged row-id set after each pass.
    std::mt19937_64 rng(424242);
    for (int pass = 0; pass < 5; ++pass) {
        for (std::size_t i = n; i > 1; --i) {
            std::uniform_int_distribution<std::size_t> pick(0, i - 1);
            table.swap_positions(static_cast<IndexPos>(i - 1),
                                 static_cast<IndexPos>(pick(rng)));
        }
        EXPECT_EQ(flagged_rowids(), expected)
            << "swap_positions dropped or misattributed a flag (carry broken)";
    }
    // The bit must track the row at every position, not just as a set.
    for (IndexPos pos = 0; pos < static_cast<IndexPos>(table.size()); ++pos) {
        const bool want = expected.count(table.row_id(pos)) > 0;
        EXPECT_EQ(table.is_flagged(pos), want) << "flag/row mismatch at pos " << pos;
    }
}

// On an eager (grid_akd) substrate the tile summaries are materialized up front
// and already include the flagged rows, so turning the index on must NOT change
// the answer: the held-out/bank machinery must stay inert for eager tiles (they
// keep outliers in non_nan with outlier_sum=0). Compare index-off vs index-on
// on the same eager grid; equality isolates "no double count / no drop" from
// any boundary or sampling artifact that the oracle comparison would conflate.
TEST(StatisticalCoverage, OutlierBankingEagerGridNoDoubleCount) {
    const std::size_t   kN = 2000;
    std::vector<double> values(kN);
    std::mt19937_64                        gen(13572468);
    std::uniform_real_distribution<double> body(1.0, 10.0);
    std::bernoulli_distribution            spike(0.01);
    for (std::size_t i = 0; i < kN; ++i) {
        values[i] = spike(gen) ? 1.0e6 : body(gen);
    }
    Dataset d("cov_banking_eager_grid", values);

    BinaryColumnStore eager(d.schema.binary_manifest_path, /*selected=*/{},
                            MeasureStorage::Eager);

    SubstrateConfig gcfg;
    gcfg.partition_size = 64;
    gcfg.partitions_per_dimension = 4;
    gcfg.data_bounds = HyperRect{{{0.0, d.xhi}, {0.0, 1.0}}};

    const RangeQuery q = d.query(0.0, d.xhi, /*rel=*/0.0);  // ForceExact ignores rel

    const auto run = [&](double budget) {
        IndexTable        table = d.make_table();
        GridAkdAccessPath grid(gcfg);
        grid.prepare(table);
        EngineConfig cfg;
        cfg.accuracy_mode = EngineConfig::AccuracyMode::ForceExact;
        cfg.persist_summaries = true;
        cfg.outlier_budget_fraction = budget;
        QueryEngine engine(eager, table, grid, cfg);
        return find(engine.execute(q, /*ordinal=*/1), AggregateOp::Sum, 0).estimate;
    };

    const double off = run(0.0);
    const double on  = run(0.02);
    const double tol = 1e-6 * std::max(1.0, std::abs(off));
    EXPECT_NEAR(on, off, tol)
        << "the index changed the eager-grid answer (double count or drop): off="
        << off << " on=" << on;
}

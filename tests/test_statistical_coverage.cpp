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
        cfg.refinement_threshold = 64;
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
                    AggregateOp op = AggregateOp::Sum) {
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

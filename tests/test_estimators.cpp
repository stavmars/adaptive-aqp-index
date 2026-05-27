// Combined-strata estimators and confidence intervals on hand-checkable data.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "a3i/aqp/estimator.hpp"
#include "a3i/aqp/query_decomposition.hpp"
#include "a3i/util/tdist.hpp"

namespace {

using namespace a3i;

ExactBucket empty_bucket(std::size_t k) {
    ExactBucket b;
    b.sum_by_measure.assign(k, 0.0);
    b.count_by_measure.assign(k, 0);
    return b;
}

// One measure, one stratum sampled at 10 of 100 with values 1..10 (all
// present): S=55, Q=385. Every expansion quantity is computed by hand below.
TEST(Estimator, SingleSampledStratumMatchesHandComputation) {
    Estimator est;
    std::vector<std::vector<StratumSample>> strata(1);
    strata[0].push_back(StratumSample{/*N=*/100, /*m=*/10, /*n=*/10,
                                      /*S=*/55.0, /*Q=*/385.0});

    auto r = est.estimate(empty_bucket(1), /*total_count=*/100, strata,
                          /*global_mean_abs=*/5.5, /*confidence=*/0.95);
    ASSERT_EQ(r.size(), 4u);  // SUM, COUNT, AVG, COUNT(*)

    const AggregateEstimate& sum = r[0];
    const AggregateEstimate& cnt = r[1];
    const AggregateEstimate& avg = r[2];
    const AggregateEstimate& cstar = r[3];

    EXPECT_EQ(sum.op, AggregateOp::Sum);
    EXPECT_NEAR(sum.estimate, 550.0, 1e-9);
    EXPECT_NEAR(sum.effective_df, 9.0, 1e-9);
    const double var_sum = 8250.0;  // 100^2 * (9.1666667/10) * 0.9
    const double crit = t_quantile(0.975, 9.0);
    EXPECT_NEAR(sum.ci_high - sum.estimate, crit * std::sqrt(var_sum), 1e-6);
    EXPECT_FALSE(sum.exact);

    EXPECT_EQ(cnt.op, AggregateOp::CountMeasure);
    EXPECT_NEAR(cnt.estimate, 100.0, 1e-9);  // all-present: Chat = N

    EXPECT_EQ(avg.op, AggregateOp::Avg);
    EXPECT_NEAR(avg.estimate, 5.5, 1e-12);

    EXPECT_EQ(cstar.op, AggregateOp::CountStar);
    EXPECT_NEAR(cstar.estimate, 100.0, 1e-12);
    EXPECT_TRUE(cstar.exact);
}

TEST(Estimator, ExactBucketOnlyIsZeroWidth) {
    Estimator est;
    ExactBucket b;
    b.sum_by_measure = {500.0};
    b.count_by_measure = {50};
    std::vector<std::vector<StratumSample>> strata(1);  // no residual strata

    auto r = est.estimate(b, /*total_count=*/60, strata, 10.0, 0.95);
    ASSERT_EQ(r.size(), 4u);
    EXPECT_TRUE(r[0].exact);
    EXPECT_NEAR(r[0].estimate, 500.0, 1e-12);
    EXPECT_DOUBLE_EQ(r[0].relative_half_width, 0.0);
    EXPECT_TRUE(r[1].exact);
    EXPECT_NEAR(r[1].estimate, 50.0, 1e-12);
    EXPECT_TRUE(r[2].exact);
    EXPECT_NEAR(r[2].estimate, 10.0, 1e-12);  // 500/50
}

TEST(Estimator, FullyReadStratumContributesExactly) {
    Estimator est;
    std::vector<std::vector<StratumSample>> strata(1);
    // m == N: the stratum is fully read; S and n are its exact totals.
    strata[0].push_back(StratumSample{/*N=*/8, /*m=*/8, /*n=*/8,
                                      /*S=*/20.0, /*Q=*/60.0});
    auto r = est.estimate(empty_bucket(1), 8, strata, 2.5, 0.95);
    EXPECT_TRUE(r[0].exact);
    EXPECT_NEAR(r[0].estimate, 20.0, 1e-12);
    EXPECT_TRUE(r[1].exact);
    EXPECT_NEAR(r[1].estimate, 8.0, 1e-12);
    EXPECT_TRUE(r[2].exact);
    EXPECT_NEAR(r[2].estimate, 2.5, 1e-12);
}

TEST(Estimator, UndersampledStratumIsUncertifiable) {
    Estimator est;
    std::vector<std::vector<StratumSample>> strata(1);
    strata[0].push_back(StratumSample{/*N=*/100, /*m=*/1, /*n=*/1,
                                      /*S=*/3.0, /*Q=*/9.0});
    auto r = est.estimate(empty_bucket(1), 100, strata, 3.0, 0.95);
    EXPECT_FALSE(r[0].exact);
    EXPECT_TRUE(std::isinf(r[0].relative_half_width));
    EXPECT_FALSE(r[1].exact);
    EXPECT_TRUE(std::isinf(r[1].relative_half_width));
}

TEST(Estimator, AvgUndefinedWhenNoNonMissingRows) {
    Estimator est;
    std::vector<std::vector<StratumSample>> strata(1);
    // Sampled rows but every one missing: n = 0 across a fully-read stratum.
    strata[0].push_back(StratumSample{/*N=*/5, /*m=*/5, /*n=*/0,
                                      /*S=*/0.0, /*Q=*/0.0});
    auto r = est.estimate(empty_bucket(1), 5, strata, 1.0, 0.95);
    EXPECT_TRUE(r[2].exact);
    EXPECT_TRUE(std::isnan(r[2].estimate));
}

TEST(Estimator, CountWilsonCenterAtBoundaryGivesPositiveVariance) {
    Estimator est;
    std::vector<std::vector<StratumSample>> strata(1);
    // All sampled rows present: p_hat = 1, so the Wilson center keeps the
    // COUNT variance strictly positive rather than zero.
    strata[0].push_back(StratumSample{/*N=*/200, /*m=*/20, /*n=*/20,
                                      /*S=*/40.0, /*Q=*/90.0});
    auto r = est.estimate(empty_bucket(1), 200, strata, 2.0, 0.95);
    EXPECT_FALSE(r[1].exact);
    EXPECT_GT(r[1].ci_high - r[1].estimate, 0.0);
}

}  // namespace

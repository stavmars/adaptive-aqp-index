// Neyman + FPC allocation: saturation, stability floor, variance-weighting,
// monotone targets, and the loop's decision-point quantities.

#include <gtest/gtest.h>

#include <vector>

#include "a3i/aqp/allocator.hpp"
#include "a3i/aqp/estimator.hpp"
#include "a3i/aqp/query_decomposition.hpp"

namespace {

using namespace a3i;

ExactBucket empty_bucket(std::size_t k) {
    ExactBucket b;
    b.sum_by_measure.assign(k, 0.0);
    b.count_by_measure.assign(k, 0);
    return b;
}

StratumAlloc one_measure(std::uint64_t N, std::uint64_t sampled, double p,
                         double mu, double sigma) {
    StratumAlloc s;
    s.N = N;
    s.sampled = sampled;
    s.priors.push_back(StratumPrior{p, mu, sigma});
    s.observed.push_back(StratumSample{N, sampled, 0, 0.0, 0.0});
    return s;
}

AccuracyTarget target(double rel) {
    AccuracyTarget t;
    t.relative_error = rel;
    t.confidence = 0.95;
    return t;
}

TEST(Allocator, LooseTargetSamplesNearStabilityFloor) {
    Allocator alloc;
    std::vector<StratumAlloc> strata = {one_measure(100000, 0, 1.0, 1.0, 1.0)};
    auto plan = alloc.plan_initial(strata, empty_bucket(1), target(0.5), 1.0);
    EXPECT_GT(plan.target[0], 10u);       // above the floor
    EXPECT_LT(plan.target[0], 100000u);   // far below exact
    EXPECT_FALSE(plan.no_target_increase);
}

TEST(Allocator, TightTargetSaturatesToExact) {
    Allocator alloc;
    std::vector<StratumAlloc> strata = {one_measure(100000, 0, 1.0, 1.0, 1.0)};
    auto plan = alloc.plan_initial(strata, empty_bucket(1), target(1e-6), 1.0);
    EXPECT_EQ(plan.target[0], 100000u);  // budget over-samples -> read all
}

TEST(Allocator, HigherVarianceStratumGetsMoreSamples) {
    Allocator alloc;
    std::vector<StratumAlloc> strata = {
        one_measure(10000, 0, 1.0, 1.0, 1.0),
        one_measure(10000, 0, 1.0, 1.0, 10.0)};
    auto plan = alloc.plan_initial(strata, empty_bucket(1), target(0.05), 1.0);
    EXPECT_GT(plan.target[1], plan.target[0]);
}

TEST(Allocator, SmallStratumIsExactifiedByFloor) {
    Allocator alloc;  // floor 10 > N
    std::vector<StratumAlloc> strata = {one_measure(5, 0, 1.0, 1.0, 1.0)};
    auto plan = alloc.plan_initial(strata, empty_bucket(1), target(0.1), 1.0);
    EXPECT_EQ(plan.target[0], 5u);  // min(floor, N) == N
}

TEST(Allocator, PlanSummaryReportsResidualAndNewReads) {
    Allocator alloc;
    std::vector<StratumAlloc> strata = {
        one_measure(1000, 0, 1.0, 1.0, 1.0),
        one_measure(2000, 0, 1.0, 1.0, 1.0)};
    auto plan = alloc.plan_initial(strata, empty_bucket(1), target(0.1), 1.0);
    EXPECT_EQ(plan.remaining_residual, 3000u);
    std::uint64_t expected = plan.target[0] + plan.target[1];  // sampled == 0
    EXPECT_EQ(plan.planned_new_reads, expected);
}

TEST(Allocator, ExactifyIfCheaperFractionExceedsHalf) {
    Allocator alloc;
    // A small, high-variance stratum whose tight target reads most of it.
    std::vector<StratumAlloc> strata = {one_measure(40, 0, 1.0, 1.0, 100.0)};
    auto plan = alloc.plan_initial(strata, empty_bucket(1), target(0.01), 1.0);
    EXPECT_GT(plan.next_round_reads_fraction(), 0.5);
}

TEST(Allocator, AdaptiveTargetsAreMonotone) {
    Allocator alloc;
    std::vector<StratumAlloc> strata = {one_measure(10000, 50, 1.0, 1.0, 1.0)};
    // Observed: 50 sampled, low variance -> plain Neyman would ask for few.
    strata[0].observed[0] = StratumSample{10000, 50, 50, 50.0, 80.0};
    std::vector<AggregateEstimate> estimates;  // empty -> tiny anticipated totals
    auto plan = alloc.plan_adaptive(strata, estimates, target(0.5), 1.0);
    EXPECT_GE(plan.target[0], 50u);  // never below the current sample count
}

TEST(Allocator, AdaptiveNoProgressIsReported) {
    Allocator alloc;
    // Already sampled to the floor with negligible variance and a loose
    // target: the planner cannot justify more reads.
    std::vector<StratumAlloc> strata = {one_measure(10000, 10, 1.0, 1.0, 0.0)};
    strata[0].observed[0] = StratumSample{10000, 10, 10, 10.0, 10.0};
    std::vector<AggregateEstimate> estimates;
    auto plan = alloc.plan_adaptive(strata, estimates, target(0.5), 1.0);
    EXPECT_TRUE(plan.no_target_increase);
}

}  // namespace

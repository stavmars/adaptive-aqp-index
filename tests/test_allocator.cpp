// Pilot-then-Neyman allocation: pilot targets, take-all of small strata,
// variance-weighted allocation under an upper confidence bound, the
// per-stratum finish rule, per-aggregate budgets (including AVG), and
// monotone clamped targets.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "a3i/aqp/allocator.hpp"
#include "a3i/aqp/estimator.hpp"
#include "a3i/aqp/query_decomposition.hpp"

namespace {

using namespace a3i;

// A stratum whose single measure has been sampled to the given state.
StratumAlloc stratum(std::uint64_t N, std::uint64_t m, std::uint64_t n,
                     double S, double Q) {
    StratumAlloc s;
    s.N = N;
    s.sampled = m;
    s.observed.push_back(StratumSample{N, m, n, S, Q});
    return s;
}

// A fresh stratum with no samples yet.
StratumAlloc fresh(std::uint64_t N) { return stratum(N, 0, 0, 0.0, 0.0); }

AccuracyTarget target(double rel) {
    AccuracyTarget t;
    t.relative_error = rel;
    t.confidence = 0.95;
    return t;
}

AggregateEstimate make_estimate(AggregateOp op, MeasureId mid, double value) {
    AggregateEstimate e;
    e.op = op;
    e.measure_id = mid;
    e.estimate = value;
    return e;
}

// Current SUM / COUNT / AVG estimates for a single-measure query.
std::vector<AggregateEstimate> estimates(double sum, double count) {
    return {make_estimate(AggregateOp::Sum, 0, sum),
            make_estimate(AggregateOp::CountMeasure, 0, count),
            make_estimate(AggregateOp::Avg, 0, count > 0 ? sum / count : 0.0)};
}

// --- Pilot plan ------------------------------------------------------------

TEST(AllocatorPilot, DrawsThePilotSizePerStratum) {
    Allocator alloc;  // pilot_sample_size = 32
    std::vector<StratumAlloc> strata = {fresh(1000), fresh(500000)};
    auto plan = alloc.plan_pilot(strata);
    EXPECT_EQ(plan.target[0], 32u);
    EXPECT_EQ(plan.target[1], 32u);
    EXPECT_FALSE(plan.no_target_increase);
}

TEST(AllocatorPilot, TakesSmallStrataWhole) {
    Allocator alloc;
    // A stratum at or below twice the pilot size is cheaper to read outright
    // than to pilot, plan, and sample in pieces.
    std::vector<StratumAlloc> strata = {fresh(50), fresh(64), fresh(65)};
    auto plan = alloc.plan_pilot(strata);
    EXPECT_EQ(plan.target[0], 50u);
    EXPECT_EQ(plan.target[1], 64u);
    EXPECT_EQ(plan.target[2], 32u);
}

TEST(AllocatorPilot, ReusedSamplesAlreadyCoverThePilot) {
    Allocator alloc;
    // A stratum that arrives with enough persisted samples needs no pilot
    // reads at all; one below the pilot size is only topped up.
    std::vector<StratumAlloc> strata = {stratum(1000, 40, 40, 40.0, 40.0),
                                        stratum(1000, 10, 10, 10.0, 10.0)};
    auto plan = alloc.plan_pilot(strata);
    EXPECT_EQ(plan.target[0], 40u);  // unchanged: no new reads
    EXPECT_EQ(plan.target[1], 32u);  // topped up to the pilot size
    EXPECT_EQ(plan.planned_new_reads, 22u);
}

// --- Neyman plan from observed statistics -----------------------------------

// Two strata with identical population and sample size; the one whose sample
// shows more spread gets the larger target.
TEST(Allocator, HigherObservedVarianceGetsMoreSamples) {
    Allocator alloc;
    // A: values tightly around 10 (Q just above S^2/m). B: values 0 or 20.
    std::vector<StratumAlloc> strata = {
        stratum(100000, 100, 100, 1000.0, 10100.0),
        stratum(100000, 100, 100, 1000.0, 20000.0)};
    auto plan = alloc.plan(strata, estimates(2.0e6, 200000), target(0.05), 10.0);
    EXPECT_GT(plan.target[1], plan.target[0]);
}

TEST(Allocator, TightTargetSaturatesToTakeAll) {
    Allocator alloc;
    std::vector<StratumAlloc> strata = {stratum(100000, 100, 100, 1000.0, 20000.0)};
    auto plan = alloc.plan(strata, estimates(1.0e6, 100000), target(1e-6), 10.0);
    EXPECT_EQ(plan.target[0], 100000u);
}

// A partial draw is never rounded up to the whole stratum: when the budget
// does not demand the full population, the plan targets only the Neyman sample
// (finish rule removed -- reuse accumulates incrementally instead, and the
// on-disk cost model reads a stratum whole only when a scan is cheaper).
TEST(Allocator, PartialDrawIsNotRoundedToFull) {
    Allocator alloc;
    std::vector<StratumAlloc> strata = {
        stratum(2000, 600, 600, 6000.0, 80000.0),
        stratum(100000, 100, 100, 1000.0, 20000.0)};
    auto plan = alloc.plan(strata, estimates(1.1e6, 102000), target(0.05), 10.0);
    // A moderate target that needs well under the full stratum must stay a
    // sample, not jump to N.
    bool any_partial = false;
    for (std::size_t h = 0; h < strata.size(); ++h) {
        EXPECT_LE(plan.target[h], strata[h].N);
        if (plan.target[h] > strata[h].sampled && plan.target[h] < strata[h].N)
            any_partial = true;
    }
    EXPECT_TRUE(any_partial) << "expected at least one stratum sampled, not taken whole";
}

// The variance upper bound shrinks with the sample size: with identical
// observed per-value statistics, the stratum that estimated them from fewer
// samples is planned more conservatively (a larger target).
TEST(Allocator, SmallerSampleGetsTheWiderVarianceBound) {
    Allocator alloc;
    // Both strata: half the values 5, half 15 (mean 10, variance ~25).
    auto obs = [](std::uint64_t m) {
        const double S = 10.0 * static_cast<double>(m);
        const double Q = 125.0 * static_cast<double>(m);
        return StratumSample{100000, m, m, S, Q};
    };
    std::vector<StratumAlloc> strata = {stratum(100000, 32, 32, 0, 0),
                                        stratum(100000, 128, 128, 0, 0)};
    strata[0].observed[0] = obs(32);
    strata[1].observed[0] = obs(128);
    auto plan = alloc.plan(strata, estimates(2.0e6, 200000), target(0.05), 10.0);
    EXPECT_GT(plan.target[0], strata[0].sampled);
    EXPECT_GT(plan.target[1], strata[1].sampled);
    EXPECT_GT(plan.target[0], plan.target[1]);
}

// All-present samples must not let the COUNT(measure) budget collapse: the
// planner uses the same boundary-adjusted presence rate as the estimator, so
// when COUNT still has a wide interval the plan keeps making progress instead
// of reporting none (which would force a needless full read).
TEST(Allocator, AllPresentSampleStillPlansForCount) {
    Allocator alloc;
    // Constant present values: the SUM variance is zero, so any progress can
    // only come from the COUNT budget.
    const double S = 7.0 * 32, Q = 49.0 * 32;
    std::vector<StratumAlloc> strata = {stratum(100000, 32, 32, S, Q)};
    auto plan = alloc.plan(strata, estimates(700000, 100000), target(0.01), 7.0);
    EXPECT_FALSE(plan.no_target_increase);
    EXPECT_GT(plan.target[0], 32u);
    EXPECT_LT(plan.target[0], 100000u);  // sampling, not a forced full read
}

// AVG can be the only failing aggregate when strata of opposite-signed means
// partially cancel in the SUM: the ratio's relative variance then exceeds both
// components'. The plan must allocate for AVG directly and make progress
// (the old behavior planned only SUM and COUNT, found both satisfied, and
// stalled into reading every stratum in full).
TEST(Allocator, AvgOnlyFailureStillMakesProgress) {
    Allocator alloc;
    // A: 100k rows, constant +10, all present. B: 20k rows, constant -8,
    // present half the time. SUM and COUNT intervals at rel 0.02 pass;
    // AVG's does not (relative half-widths ~0.012 / 0.017 / 0.027).
    std::vector<StratumAlloc> strata = {
        stratum(100000, 200, 200, 2000.0, 20000.0),
        stratum(20000, 200, 100, -800.0, 6400.0)};
    const double t_sum = 100000.0 * 10.0 - 20000.0 * 0.5 * 8.0;  // 920000
    const double t_count = 100000.0 + 10000.0;
    auto plan = alloc.plan(strata, estimates(t_sum, t_count), target(0.02), 10.0);
    EXPECT_FALSE(plan.no_target_increase);
    // The AVG noise lives entirely in stratum B; stratum A's constant values
    // contribute nothing, so the plan must not grow A.
    EXPECT_EQ(plan.target[0], strata[0].sampled);
    EXPECT_GT(plan.target[1], strata[1].sampled);
}

TEST(Allocator, TargetsAreMonotoneAndClamped) {
    Allocator alloc;
    std::vector<StratumAlloc> strata = {stratum(10000, 50, 50, 500.0, 5050.0)};
    auto plan = alloc.plan(strata, estimates(100000, 10000), target(0.5), 10.0);
    EXPECT_GE(plan.target[0], 50u);
    EXPECT_LE(plan.target[0], 10000u);
}

TEST(Allocator, SatisfiedBudgetsPlanNoReads) {
    Allocator alloc;
    // A large, thoroughly sampled, low-variance stratum under a loose target:
    // every budget is already met, so the plan holds (the loop only consults
    // the plan when some aggregate still fails).
    std::vector<StratumAlloc> strata = {
        stratum(100000, 5000, 5000, 50000.0, 505000.0)};
    auto plan = alloc.plan(strata, estimates(1.0e6, 100000), target(0.5), 10.0);
    EXPECT_TRUE(plan.no_target_increase);
    EXPECT_EQ(plan.target[0], 5000u);
}

TEST(Allocator, PlanReportsResidualAndNewReads) {
    Allocator alloc;
    std::vector<StratumAlloc> strata = {fresh(1000), fresh(2000)};
    auto plan = alloc.plan_pilot(strata);
    EXPECT_EQ(plan.remaining_residual, 3000u);
    EXPECT_EQ(plan.planned_new_reads, plan.target[0] + plan.target[1]);
}

}  // namespace

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "a3i/aqp/summary.hpp"

using namespace a3i;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Brute-force reference moments over a value set, ignoring NaN.
struct RefMoments {
    std::uint64_t n = 0;
    double sum = 0.0, sum_sq = 0.0;
    void add(double x) {
        if (std::isnan(x)) return;
        ++n;
        sum += x;
        sum_sq += x * x;
    }
    double variance() const {
        if (n < 2) return 0.0;
        return (sum_sq - sum * sum / static_cast<double>(n)) /
               static_cast<double>(n - 1);
    }
};

}  // namespace

TEST(MomentStats, MatchesBruteForceSumSumSqVariance) {
    const std::vector<double> values{3.0, -1.5, 7.25, 0.0, 12.0, -4.0, 5.5};
    MomentStats ms;
    RefMoments ref;
    for (double v : values) {
        ms.add_if_present(v);
        ref.add(v);
    }
    EXPECT_EQ(ms.non_nan_count, ref.n);
    EXPECT_NEAR(ms.sum(), ref.sum, 1e-9);
    EXPECT_NEAR(ms.sum_sq(), ref.sum_sq, 1e-9);
    EXPECT_NEAR(ms.sample_variance(), ref.variance(), 1e-9);
}

TEST(MomentStats, NaNIsIgnoredAsMissing) {
    MomentStats ms;
    ms.add_if_present(2.0);
    ms.add_if_present(kNaN);
    ms.add_if_present(4.0);
    ms.add_if_present(kNaN);
    EXPECT_EQ(ms.non_nan_count, 2u);
    EXPECT_NEAR(ms.sum(), 6.0, 1e-12);
    EXPECT_NEAR(ms.sum_sq(), 20.0, 1e-12);
}

TEST(MomentStats, VarianceZeroBelowTwoObservations) {
    MomentStats empty;
    EXPECT_DOUBLE_EQ(empty.sample_variance(), 0.0);
    MomentStats one;
    one.add_if_present(42.0);
    EXPECT_DOUBLE_EQ(one.sample_variance(), 0.0);
}

TEST(MomentStats, MergeEqualsSingleAccumulation) {
    const std::vector<double> all{1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0};
    MomentStats whole;
    for (double v : all) whole.add_if_present(v);

    MomentStats a, b;
    for (std::size_t i = 0; i < all.size(); ++i) {
        (i < 3 ? a : b).add_if_present(all[i]);
    }
    a.merge(b);

    EXPECT_EQ(a.non_nan_count, whole.non_nan_count);
    EXPECT_NEAR(a.sum(), whole.sum(), 1e-9);
    EXPECT_NEAR(a.sum_sq(), whole.sum_sq(), 1e-9);
    EXPECT_NEAR(a.sample_variance(), whole.sample_variance(), 1e-9);
}

TEST(MomentStats, MergeWithEmptyIsIdentity) {
    MomentStats a;
    a.add_if_present(5.0);
    a.add_if_present(9.0);
    const double before_sum = a.sum();

    MomentStats empty;
    a.merge(empty);
    EXPECT_EQ(a.non_nan_count, 2u);
    EXPECT_NEAR(a.sum(), before_sum, 1e-12);

    MomentStats c;
    c.merge(a);  // empty.merge(non-empty) copies
    EXPECT_EQ(c.non_nan_count, 2u);
    EXPECT_NEAR(c.sum(), before_sum, 1e-12);
}

TEST(SampleTracker, AddIsIdempotentAndCounts) {
    SampleTracker t(100);
    EXPECT_EQ(t.size(), 100u);
    EXPECT_EQ(t.count(), 0u);
    EXPECT_FALSE(t.contains(7));

    t.add(7);
    t.add(7);  // idempotent
    t.add(63);
    t.add(64);  // crosses a word boundary
    EXPECT_TRUE(t.contains(7));
    EXPECT_TRUE(t.contains(63));
    EXPECT_TRUE(t.contains(64));
    EXPECT_FALSE(t.contains(8));
    EXPECT_EQ(t.count(), 3u);
}

TEST(SampleTracker, OutOfRangeAddThrowsAndContainsIsFalse) {
    SampleTracker t(10);
    EXPECT_FALSE(t.contains(10));
    EXPECT_FALSE(t.contains(1000));
    EXPECT_THROW(t.add(10), std::out_of_range);
}

TEST(MeasureSummary, StateInferredFromCounts) {
    MeasureSummary s;
    s.population_size = 0;
    s.sampled_rows = 0;
    EXPECT_FALSE(s.sampled());
    EXPECT_FALSE(s.complete());  // absent

    s.population_size = 10;
    s.sampled_rows = 0;
    EXPECT_FALSE(s.sampled());   // absent
    EXPECT_FALSE(s.complete());

    s.sampled_rows = 4;
    EXPECT_TRUE(s.sampled());    // sampled
    EXPECT_FALSE(s.complete());

    s.sampled_rows = 10;
    EXPECT_FALSE(s.sampled());
    EXPECT_TRUE(s.complete());   // exact

    s.sampled_rows = 12;         // over-read still complete
    EXPECT_TRUE(s.complete());
}

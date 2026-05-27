// Student-t and normal quantiles checked against published table values.

#include <gtest/gtest.h>

#include <cmath>

#include "a3i/util/tdist.hpp"

namespace {

using namespace a3i;

TEST(NormalQuantile, KnownPercentiles) {
    EXPECT_NEAR(normal_quantile(0.975), 1.959963985, 1e-6);
    EXPECT_NEAR(normal_quantile(0.95), 1.644853627, 1e-6);
    EXPECT_NEAR(normal_quantile(0.5), 0.0, 1e-9);
    EXPECT_NEAR(normal_quantile(0.025), -1.959963985, 1e-6);
    EXPECT_NEAR(normal_quantile(0.99), 2.326347874, 1e-6);
}

TEST(TQuantile, UpperTailTableValues) {
    // Two-sided 95% critical values (upper 0.975 quantile).
    EXPECT_NEAR(t_quantile(0.975, 1.0), 12.7062, 1e-3);
    EXPECT_NEAR(t_quantile(0.975, 2.0), 4.30265, 1e-3);
    EXPECT_NEAR(t_quantile(0.975, 5.0), 2.57058, 1e-3);
    EXPECT_NEAR(t_quantile(0.975, 10.0), 2.22814, 1e-3);
    EXPECT_NEAR(t_quantile(0.975, 30.0), 2.04227, 1e-3);
    EXPECT_NEAR(t_quantile(0.975, 100.0), 1.98397, 1e-3);
}

TEST(TQuantile, OtherConfidences) {
    EXPECT_NEAR(t_quantile(0.95, 10.0), 1.81246, 1e-3);
    EXPECT_NEAR(t_quantile(0.99, 5.0), 3.36493, 1e-3);
    EXPECT_NEAR(t_quantile(0.995, 20.0), 2.84534, 1e-3);
}

TEST(TQuantile, LargeDfApproachesNormal) {
    EXPECT_NEAR(t_quantile(0.975, 1.0e9), 1.959963985, 1e-4);
    EXPECT_NEAR(t_quantile(0.975, 1.0e8), 1.959963985, 1e-4);
}

TEST(TQuantile, SymmetryAndMonotonicity) {
    EXPECT_NEAR(t_quantile(0.025, 10.0), -t_quantile(0.975, 10.0), 1e-9);
    EXPECT_NEAR(t_quantile(0.5, 7.0), 0.0, 1e-9);
    // Wider tails at fewer degrees of freedom.
    EXPECT_GT(t_quantile(0.975, 2.0), t_quantile(0.975, 50.0));
}

TEST(TQuantile, FractionalDegreesOfFreedom) {
    // Welch-Satterthwaite yields non-integer df; the value must lie between
    // the neighbouring integer-df critical values.
    const double v = t_quantile(0.975, 7.5);
    EXPECT_LT(v, t_quantile(0.975, 7.0));
    EXPECT_GT(v, t_quantile(0.975, 8.0));
}

}  // namespace

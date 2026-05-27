// Combined-strata expansion estimators with confidence intervals.
//
// A query's qualifying objects are split into disjoint strata plus a
// deterministic exact bucket. Each residual stratum is a simple random
// sample without replacement of its population; its expansion estimates and
// finite-population-corrected variances add across strata, and the exact
// bucket adds with zero variance. From the combined totals and variance
// components this produces, per measure, the SUM / COUNT(measure) / AVG
// estimates and, once, the always-exact COUNT(*).
//
// The reported interval uses a Student-t critical value whose degrees of
// freedom come from a Satterthwaite approximation of the variance-component
// sum, so a few heavy strata widen the interval and many well-sampled strata
// drive it toward the normal value. Magnitude floors keep the relative
// half-width meaningful when an estimate is numerically tiny.

#pragma once

#include <cstdint>
#include <vector>

#include "a3i/aqp/query_decomposition.hpp"
#include "a3i/core/query.hpp"
#include "a3i/core/types.hpp"

namespace a3i {

/// One residual stratum's current sample state for a single measure.
///   N - population (in-stratum objects, including missing)
///   m - rows sampled so far (including missing)
///   n - non-missing rows among the sampled
///   S - sum of the non-missing sampled values
///   Q - sum of squares of the non-missing sampled values
/// A stratum with `m >= N` is fully read and contributes exactly.
struct StratumSample {
    std::uint64_t N = 0;
    std::uint64_t m = 0;
    std::uint64_t n = 0;
    double        S = 0.0;
    double        Q = 0.0;
};

class Estimator {
public:
    /// Produce one estimate per (measure x {SUM, COUNT(measure), AVG}) in
    /// measure order, followed by a single exact COUNT(*).
    ///
    /// `strata_by_measure[mid]` lists the residual strata contributing to
    /// measure `mid`; `exact_bucket` holds the deterministic totals already
    /// known exactly; `total_count` is the qualifying-object count.
    /// `global_mean_abs` is the maximum absolute per-measure global mean, used
    /// only to floor the relative-half-width denominator. `confidence` sets
    /// the two-sided interval level.
    std::vector<AggregateEstimate> estimate(
        const ExactBucket& exact_bucket, std::uint64_t total_count,
        const std::vector<std::vector<StratumSample>>& strata_by_measure,
        double global_mean_abs, double confidence) const;
};

}  // namespace a3i

// Per-(partition, measure) aggregate state and its building blocks.
//
// A summary records what has been learned about one measure over one
// partition's object population: how many objects it covers, how many
// have been sampled so far, and the running moments of the non-missing
// sampled values. The three logical states an index reasons about --
// absent, sampled, exact -- are not stored as an enum; they are inferred
// from the counts (no summary => absent; some-but-not-all sampled =>
// sampled; all sampled => exact).

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "a3i/core/types.hpp"

namespace a3i {

/// Running first/second moments over the NON-missing (non-NaN) values seen
/// so far, accumulated numerically stably (Welford). Estimator inputs are
/// taken only through sum()/sum_sq(); raw running squares are never stored.
struct MomentStats {
    std::uint64_t non_nan_count = 0;
    double mean = 0.0;
    double m2   = 0.0;  // sum of squared deviations from the running mean

    /// Fold one value in; NaN is treated as missing and ignored.
    void add_if_present(double x);

    /// S = sum of the non-missing values.
    double sum() const { return mean * static_cast<double>(non_nan_count); }

    /// Q = sum of squares of the non-missing values.
    double sum_sq() const {
        return m2 + mean * mean * static_cast<double>(non_nan_count);
    }

    /// Unbiased sample variance, guarded to be >= 0. Zero when fewer than
    /// two values have been seen (not certifiable from one observation).
    double sample_variance() const;

    /// Combine another disjoint set of observations into this one
    /// (numerically stable parallel form).
    void merge(const MomentStats& other);
};

/// Bitset marking which stratum-local positions have already been sampled,
/// so cumulative without-replacement sampling never draws a position twice.
/// Positions are stratum-local (offset from the stratum's begin).
class SampleTracker {
public:
    explicit SampleTracker(std::uint64_t stratum_size);

    bool          contains(IndexPos local) const;
    void          add(IndexPos local);
    std::uint64_t count() const noexcept { return count_; }
    std::uint64_t size()  const noexcept { return size_; }

private:
    std::vector<std::uint64_t> bits_;
    std::uint64_t              count_ = 0;
    std::uint64_t              size_  = 0;
};

/// One round's worth of newly sampled rows for a single measure, ready to
/// fold into a summary: the number of rows newly drawn (including missing
/// ones) and the moments of their non-missing values.
struct SampleDelta {
    std::uint64_t new_sampled_rows = 0;
    MomentStats   moments;
};

/// Aggregate state of one (partition, measure) pair. State is inferred:
///   absent  -> sampled_rows == 0
///   sampled -> 0 < sampled_rows < population_size
///   exact   -> sampled_rows >= population_size (> 0)
struct MeasureSummary {
    std::uint64_t population_size = 0;  // objects in the partition, incl. missing
    std::uint64_t sampled_rows    = 0;  // body rows drawn (never incl. held-out rows)
    MomentStats   non_nan;              // moments over the non-missing sampled values
    std::shared_ptr<SampleTracker> tracker;  // shared across the partition's measures

    // Held-out rows banked exactly. These rows are excluded from the sample, so
    // their values are kept here, separate from non_nan, and the body's
    // sampled_rows/population accounting works against the body only
    // (population_size - outlier_rows). outlier_rows and outliers_materialized
    // are partition-wide facts replicated onto each measure slot, like
    // population_size; outlier_sum/outlier_count are per-measure. Zero/false on
    // every path that holds nothing out (including a fully materialized summary
    // whose held-out rows are already inside non_nan, which leaves outlier_rows
    // at 0).
    double        outlier_sum   = 0.0;
    std::uint64_t outlier_count = 0;
    std::uint64_t outlier_rows  = 0;
    bool          outliers_materialized = false;

    bool sampled() const noexcept {
        return sampled_rows > 0 && sampled_rows < population_size - outlier_rows;
    }
    bool complete() const noexcept {
        return population_size > 0
            && sampled_rows >= population_size - outlier_rows
            && (outlier_rows == 0 || outliers_materialized);
    }
};

}  // namespace a3i

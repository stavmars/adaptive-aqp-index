// Stratified Neyman allocation with finite-population correction.
//
// Given the residual strata of a query and a relative-error target, the
// allocator sets a cumulative sample target per stratum. Round one plans from
// priors (global per-measure statistics); later rounds replan from each
// stratum's observed sample statistics. Allocation is by the closed-form
// Neyman solution under a variance budget derived from the target, iterated
// to saturate strata that the budget would over-sample (those become exact),
// then floored so every sampling stratum is large enough to estimate its own
// variance. Targets are monotone: a stratum's cumulative target never falls
// below its previous value or its current sample count.
//
// The plan also reports the two quantities the adaptive loop needs to decide
// between sampling and exactification: how many new reads the next round would
// cost relative to the residual that remains, and whether the plan raised any
// target at all.

#pragma once

#include <cstdint>
#include <vector>

#include "a3i/aqp/estimator.hpp"
#include "a3i/aqp/query_decomposition.hpp"
#include "a3i/core/query.hpp"

namespace a3i {

struct AllocatorConfig {
    std::uint64_t stability_sample_floor = 10;
    double        exactification_sample_fraction = 0.5;
    std::uint64_t max_sampling_rounds = 10;
};

/// Prior moments for one (stratum, measure) used in round-one planning.
///   p        - fraction of non-missing values expected
///   mu_nn    - mean of the non-missing values
///   sigma_nn - standard deviation of the non-missing values
struct StratumPrior {
    double p = 0.0;
    double mu_nn = 0.0;
    double sigma_nn = 0.0;
};

/// One residual stratum for allocation. `sampled` is the partition-wide
/// cumulative sample count (measures are sampled in lockstep). `priors` are
/// used in round one; `observed` carries each measure's current sample stats
/// for later rounds (ignored when a measure has fewer than two samples).
struct StratumAlloc {
    std::uint64_t              N = 0;
    std::uint64_t              sampled = 0;
    std::vector<StratumPrior>  priors;
    std::vector<StratumSample> observed;
};

struct AllocationPlan {
    std::vector<std::uint64_t> target;  // cumulative target per stratum
    std::uint64_t planned_new_reads = 0;
    std::uint64_t remaining_residual = 0;
    bool          no_target_increase = true;

    /// Fraction of the remaining residual the next sampling round would read.
    double next_round_reads_fraction() const {
        return remaining_residual == 0
                   ? 0.0
                   : static_cast<double>(planned_new_reads) /
                         static_cast<double>(remaining_residual);
    }
};

class Allocator {
public:
    explicit Allocator(AllocatorConfig cfg = {}) : cfg_(cfg) {}

    const AllocatorConfig& config() const noexcept { return cfg_; }

    /// Round-one plan from priors only.
    AllocationPlan plan_initial(const std::vector<StratumAlloc>& strata,
                                const ExactBucket& exact_bucket,
                                const AccuracyTarget& target,
                                double global_mean_abs) const;

    /// Round 2+ plan from observed per-stratum statistics and the current
    /// estimates (used as the anticipated totals).
    AllocationPlan plan_adaptive(const std::vector<StratumAlloc>& strata,
                                 const std::vector<AggregateEstimate>& estimates,
                                 const AccuracyTarget& target,
                                 double global_mean_abs) const;

private:
    AllocatorConfig cfg_;
};

}  // namespace a3i

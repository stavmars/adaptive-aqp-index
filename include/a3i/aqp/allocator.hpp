// Pilot-then-Neyman sample allocation with finite-population correction.
//
// The loop sizes its reads in two phases:
//
//  * `plan_pilot` raises every residual stratum to a small fixed pilot
//    sample so its variance can be estimated from its own rows. A stratum
//    already holding that many persisted samples from earlier queries needs
//    no pilot reads; a stratum too small to be worth piloting (at or below
//    twice the pilot size) is read whole instead.
//  * `plan` re-solves the closed-form Neyman allocation under a per-aggregate
//    variance budget derived from the accuracy target, using each stratum's
//    OBSERVED statistics. Three budgets are solved per measure -- SUM, COUNT
//    (with the same boundary-adjusted presence rate the estimator uses), and
//    AVG (on the variance of the ratio-linearized residuals, the standard
//    allocation for a ratio estimate) -- and a stratum's target is the
//    maximum demand across them.
//
// Two conservatism devices keep the plan honest about what a finite sample
// can know. Observed variances enter through an upper confidence bound (the
// chi-squared bound on a sample variance), so an early sample that happens to
// look low variance does not under-size the next round. And the allocation is
// iterated to saturation: a stratum the budget would over-sample is read in
// full ("take-all"), removed from the pool, and the rest re-solved -- the
// same rule extends to strata where the planned draw would exceed a fixed
// fraction of the rows remaining, where finishing the stratum costs little
// more and yields an exact, reusable summary.

#pragma once

#include <cstdint>
#include <vector>

#include "a3i/aqp/estimator.hpp"
#include "a3i/aqp/query_decomposition.hpp"
#include "a3i/core/query.hpp"

namespace a3i {

struct AllocatorConfig {
    /// Per-stratum first-phase sample: enough rows for a usable variance
    /// estimate (a few dozen; below that the chi-squared bound on the
    /// variance becomes too loose to plan with). Strata at or below twice
    /// this size are read whole instead of piloted.
    std::uint64_t pilot_sample_size = 32;
    /// Total read rounds per query, counting the pilot and the terminal
    /// full read of the residual if one is needed.
    std::uint64_t max_sampling_rounds = 4;
};

/// One residual stratum for allocation. `sampled` is the partition-wide
/// cumulative sample count (measures are sampled in lockstep); `observed`
/// carries each measure's current sample statistics.
struct StratumAlloc {
    std::uint64_t              N = 0;
    std::uint64_t              sampled = 0;
    std::vector<StratumSample> observed;
};

struct AllocationPlan {
    std::vector<std::uint64_t> target;  // cumulative target per stratum
    std::uint64_t planned_new_reads = 0;
    std::uint64_t remaining_residual = 0;
    bool          no_target_increase = true;

    /// Fraction of the remaining residual the next round would read.
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

    /// First phase: raise every stratum to the pilot size (taking small
    /// strata whole), so the second phase can plan from observed statistics.
    AllocationPlan plan_pilot(const std::vector<StratumAlloc>& strata) const;

    /// Second phase (and any corrective re-plan): Neyman + FPC from observed
    /// per-stratum statistics, with the current `estimates` supplying the
    /// anticipated totals that scale each aggregate's variance budget.
    AllocationPlan plan(const std::vector<StratumAlloc>& strata,
                        const std::vector<AggregateEstimate>& estimates,
                        const AccuracyTarget& target,
                        double global_mean_abs) const;

private:
    AllocatorConfig cfg_;
};

}  // namespace a3i

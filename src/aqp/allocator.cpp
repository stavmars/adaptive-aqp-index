#include "a3i/aqp/allocator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace a3i {

namespace {

constexpr double kMagnitudeFloorDelta = 1e-6;
constexpr double kPlanZ = 1.959963984540054;  // plan with 1.96, report with t

// Per (stratum, measure) inputs to the closed-form allocation.
struct Eff {
    double sh_sum = 0.0;        // prior/observed stdev of the null-as-zero value
    double sh_count = 0.0;      // prior/observed stdev of the presence indicator
    double contrib_sum = 0.0;   // anticipated SUM contribution (N * p * mu)
    double contrib_count = 0.0; // anticipated COUNT contribution (N * p)
};

Eff eff_from_prior(std::uint64_t N, const StratumPrior& pr) {
    Eff e;
    const double p = pr.p;
    e.sh_sum = std::sqrt(std::max(
        0.0, p * pr.sigma_nn * pr.sigma_nn + p * (1.0 - p) * pr.mu_nn * pr.mu_nn));
    e.sh_count = std::sqrt(std::max(0.0, p * (1.0 - p)));
    e.contrib_sum = static_cast<double>(N) * p * pr.mu_nn;
    e.contrib_count = static_cast<double>(N) * p;
    return e;
}

Eff eff_from_observed(const StratumSample& s) {
    Eff e;
    const double m = static_cast<double>(s.m);
    const double n = static_cast<double>(s.n);
    const double p = n / m;
    double s2z = (s.Q - s.S * s.S / m) / (m - 1.0);
    if (s2z < 0.0) s2z = 0.0;
    e.sh_sum = std::sqrt(s2z);
    e.sh_count = std::sqrt(std::max(0.0, p * (1.0 - p)));
    e.contrib_sum = static_cast<double>(s.N) * s.S / m;
    e.contrib_count = static_cast<double>(s.N) * p;
    return e;
}

// Closed-form Neyman+FPC sample sizes for a single aggregate under a variance
// budget V*, iterated so strata the budget would over-sample saturate to N_h.
std::vector<std::uint64_t> neyman_alloc(const std::vector<std::uint64_t>& N,
                                        const std::vector<double>& sh,
                                        double v_star) {
    const std::size_t H = N.size();
    std::vector<std::uint64_t> n(H, 0);
    std::vector<bool> saturated(H, false);

    for (std::size_t iter = 0; iter <= H + 1; ++iter) {
        double A = 0.0, B = 0.0;
        for (std::size_t h = 0; h < H; ++h) {
            if (saturated[h]) continue;
            const double W = static_cast<double>(N[h]) * sh[h];
            A += W;
            B += static_cast<double>(N[h]) * sh[h] * sh[h];
        }
        if (A <= 0.0) break;  // no variance left among active strata

        const double n_tot = A * A / (v_star + B);
        bool changed = false;
        for (std::size_t h = 0; h < H; ++h) {
            if (saturated[h]) continue;
            const double W = static_cast<double>(N[h]) * sh[h];
            const double nh = std::ceil(n_tot * W / A);
            if (nh >= static_cast<double>(N[h])) {
                n[h] = N[h];
                saturated[h] = true;
                changed = true;
            } else {
                n[h] = static_cast<std::uint64_t>(nh < 0.0 ? 0.0 : nh);
            }
        }
        if (!changed) break;
    }
    return n;
}

}  // namespace

AllocationPlan Allocator::plan_initial(const std::vector<StratumAlloc>& strata,
                                       const ExactBucket& exact_bucket,
                                       const AccuracyTarget& target,
                                       double global_mean_abs) const {
    const std::size_t H = strata.size();
    const std::size_t k = H == 0 ? 0 : strata[0].priors.size();

    std::vector<std::uint64_t> N(H);
    std::uint64_t sum_population = 0;
    for (std::size_t h = 0; h < H; ++h) {
        N[h] = strata[h].N;
        sum_population += strata[h].N;
    }
    const double tau_sum =
        kMagnitudeFloorDelta * global_mean_abs * static_cast<double>(sum_population);
    const double tau_count =
        std::max(1.0, kMagnitudeFloorDelta * static_cast<double>(sum_population));

    std::vector<std::uint64_t> target_n(H, 0);
    const double rel = target.relative_error;

    for (std::size_t mid = 0; mid < k; ++mid) {
        std::vector<double> sh_sum(H), sh_count(H);
        double t_sum = mid < exact_bucket.sum_by_measure.size()
                           ? exact_bucket.sum_by_measure[mid]
                           : 0.0;
        double t_count = mid < exact_bucket.count_by_measure.size()
                             ? static_cast<double>(exact_bucket.count_by_measure[mid])
                             : 0.0;
        for (std::size_t h = 0; h < H; ++h) {
            const Eff e = eff_from_prior(strata[h].N, strata[h].priors[mid]);
            sh_sum[h] = e.sh_sum;
            sh_count[h] = e.sh_count;
            t_sum += e.contrib_sum;
            t_count += e.contrib_count;
        }
        const double t_ant_sum = std::max(std::abs(t_sum), tau_sum);
        const double t_ant_count = std::max(t_count, tau_count);
        const double v_sum =
            rel > 0.0 ? std::pow(rel * t_ant_sum / kPlanZ, 2.0) : 0.0;
        const double v_count =
            rel > 0.0 ? std::pow(rel * t_ant_count / kPlanZ, 2.0) : 0.0;

        const auto n_sum = neyman_alloc(N, sh_sum, v_sum);
        const auto n_count = neyman_alloc(N, sh_count, v_count);
        for (std::size_t h = 0; h < H; ++h) {
            target_n[h] = std::max({target_n[h], n_sum[h], n_count[h]});
        }
    }

    AllocationPlan plan;
    plan.target.resize(H);
    for (std::size_t h = 0; h < H; ++h) {
        std::uint64_t t = target_n[h];
        // Stability floor: every non-empty stratum reaches the floor or is
        // exactified when it has fewer eligible rows than the floor.
        t = std::max(t, std::min(cfg_.stability_sample_floor, strata[h].N));
        t = std::min(t, strata[h].N);
        t = std::max(t, strata[h].sampled);
        plan.target[h] = t;
        plan.remaining_residual += strata[h].N - strata[h].sampled;
        if (t > strata[h].sampled) {
            plan.planned_new_reads += t - strata[h].sampled;
            plan.no_target_increase = false;
        }
    }
    return plan;
}

AllocationPlan Allocator::plan_adaptive(
    const std::vector<StratumAlloc>& strata,
    const std::vector<AggregateEstimate>& estimates,
    const AccuracyTarget& target, double global_mean_abs) const {
    const std::size_t H = strata.size();
    const std::size_t k = H == 0 ? 0 : strata[0].observed.size();

    std::vector<std::uint64_t> N(H);
    std::uint64_t sum_population = 0;
    for (std::size_t h = 0; h < H; ++h) {
        N[h] = strata[h].N;
        sum_population += strata[h].N;
    }
    const double tau_sum =
        kMagnitudeFloorDelta * global_mean_abs * static_cast<double>(sum_population);
    const double tau_count =
        std::max(1.0, kMagnitudeFloorDelta * static_cast<double>(sum_population));

    // Index the current estimates by (op, measure) for the anticipated totals.
    auto estimate_for = [&](AggregateOp op, MeasureId mid) -> double {
        for (const AggregateEstimate& e : estimates) {
            if (e.op == op && e.measure_id == mid) return e.estimate;
        }
        return 0.0;
    };

    std::vector<std::uint64_t> target_n(H, 0);
    const double rel = target.relative_error;

    for (std::size_t mid = 0; mid < k; ++mid) {
        std::vector<double> sh_sum(H), sh_count(H);
        for (std::size_t h = 0; h < H; ++h) {
            const StratumSample& obs = strata[h].observed[mid];
            const Eff e = obs.m >= 2
                              ? eff_from_observed(obs)
                              : (mid < strata[h].priors.size()
                                     ? eff_from_prior(strata[h].N,
                                                      strata[h].priors[mid])
                                     : Eff{});
            sh_sum[h] = e.sh_sum;
            sh_count[h] = e.sh_count;
        }
        const double t_ant_sum = std::max(
            std::abs(estimate_for(AggregateOp::Sum, static_cast<MeasureId>(mid))),
            tau_sum);
        const double t_ant_count =
            std::max(std::abs(estimate_for(AggregateOp::CountMeasure,
                                           static_cast<MeasureId>(mid))),
                     tau_count);
        const double v_sum =
            rel > 0.0 ? std::pow(rel * t_ant_sum / kPlanZ, 2.0) : 0.0;
        const double v_count =
            rel > 0.0 ? std::pow(rel * t_ant_count / kPlanZ, 2.0) : 0.0;

        const auto n_sum = neyman_alloc(N, sh_sum, v_sum);
        const auto n_count = neyman_alloc(N, sh_count, v_count);
        for (std::size_t h = 0; h < H; ++h) {
            target_n[h] = std::max({target_n[h], n_sum[h], n_count[h]});
        }
    }

    AllocationPlan plan;
    plan.target.resize(H);
    for (std::size_t h = 0; h < H; ++h) {
        std::uint64_t t = target_n[h];
        t = std::max(t, std::min(cfg_.stability_sample_floor, strata[h].N));
        t = std::min(t, strata[h].N);
        // Monotone: never below the previous sample count.
        t = std::max(t, strata[h].sampled);
        plan.target[h] = t;
        plan.remaining_residual += strata[h].N - strata[h].sampled;
        if (t > strata[h].sampled) {
            plan.planned_new_reads += t - strata[h].sampled;
            plan.no_target_increase = false;
        }
    }
    return plan;
}

}  // namespace a3i

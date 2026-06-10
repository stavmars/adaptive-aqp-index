#include "a3i/aqp/allocator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "a3i/util/tdist.hpp"

namespace a3i {

namespace {

constexpr double kMagnitudeFloorDelta = 1e-6;

// Multiply an observed variance by this factor to get its one-sided upper
// confidence bound: (m-1) s^2 / chi^2_{delta, m-1} bounds the true variance
// from above with probability 1-delta under approximate normality. Capped so
// a freak tiny-sample draw cannot explode a plan; for the pilot sizes the
// planner guarantees (>= 32 samples) the factor sits around 1.6 and decays
// toward 1 as the sample grows.
double variance_bound_factor(std::uint64_t m, double delta) {
    if (m < 2) return 1.0;
    const double nu = static_cast<double>(m - 1);
    const double f = nu / chi_squared_quantile(delta, nu);
    return std::min(std::max(f, 1.0), 4.0);
}

// Per (stratum, measure) standard deviations entering the three budgets.
struct Spread {
    double sh_sum = 0.0;  // null-as-zero value
    double sh_count = 0.0;  // presence indicator
    double sh_avg = 0.0;  // ratio-linearized residual z - mu * c
};

// Observed spreads under the variance upper bound. `mu` is the current AVG
// estimate for this measure (NaN when undefined); the residual variance for
// the AVG budget linearizes the ratio at it.
Spread spread_from_observed(const StratumSample& s, double mu, double delta) {
    Spread out;
    if (s.m < 2 || s.m >= s.N) return out;  // nothing to plan, or already exact

    const double m = static_cast<double>(s.m);
    const double n = static_cast<double>(s.n);
    const double bound = variance_bound_factor(s.m, delta);

    double s2z = (s.Q - s.S * s.S / m) / (m - 1.0);
    if (s2z < 0.0) s2z = 0.0;
    out.sh_sum = std::sqrt(s2z * bound);

    const double p_adj = presence_rate_for_variance(s.n, s.m);
    out.sh_count = std::sqrt(std::max(0.0, p_adj * (1.0 - p_adj)));

    if (std::isfinite(mu)) {
        const double sum_e = s.S - mu * n;                       // sum of residuals
        const double sum_e2 = s.Q - 2.0 * mu * s.S + mu * mu * n;  // their squares
        double s2e = (sum_e2 - sum_e * sum_e / m) / (m - 1.0);
        if (s2e < 0.0) s2e = 0.0;
        out.sh_avg = std::sqrt(s2e * bound);
    }
    return out;
}

// Closed-form Neyman+FPC sample sizes for one aggregate under a variance
// budget V*, iterated so strata the budget would over-sample saturate to N_h
// (the take-all strata of survey practice) and the rest re-solve.
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

AllocationPlan Allocator::plan_pilot(
    const std::vector<StratumAlloc>& strata) const {
    const std::uint64_t m0 = cfg_.pilot_sample_size;
    AllocationPlan plan;
    plan.target.resize(strata.size());
    for (std::size_t h = 0; h < strata.size(); ++h) {
        const std::uint64_t N = strata[h].N;
        // Piloting a stratum one could nearly read outright is wasted
        // bookkeeping; at or below twice the pilot size, take it whole.
        std::uint64_t t = N <= 2 * m0 ? N : m0;
        t = std::max(t, strata[h].sampled);  // persisted samples already count
        plan.target[h] = t;
        plan.remaining_residual += N - strata[h].sampled;
        if (t > strata[h].sampled) {
            plan.planned_new_reads += t - strata[h].sampled;
            plan.no_target_increase = false;
        }
    }
    return plan;
}

AllocationPlan Allocator::plan(const std::vector<StratumAlloc>& strata,
                               const std::vector<AggregateEstimate>& estimates,
                               const AccuracyTarget& target,
                               double global_mean_abs) const {
    const std::size_t H = strata.size();
    const std::size_t k = H == 0 ? 0 : strata[0].observed.size();

    std::vector<std::uint64_t> N(H);
    std::uint64_t sum_population = 0;
    for (std::size_t h = 0; h < H; ++h) {
        N[h] = strata[h].N;
        sum_population += strata[h].N;
    }
    // Magnitude floors guard the budgets when an estimate is numerically
    // tiny, exactly as the estimator floors its relative half-widths.
    const double tau_sum = kMagnitudeFloorDelta * global_mean_abs *
                           static_cast<double>(sum_population);
    const double tau_count =
        std::max(1.0, kMagnitudeFloorDelta * static_cast<double>(sum_population));
    const double tau_avg = kMagnitudeFloorDelta * global_mean_abs;

    auto estimate_for = [&](AggregateOp op, MeasureId mid) -> double {
        for (const AggregateEstimate& e : estimates) {
            if (e.op == op && e.measure_id == mid) return e.estimate;
        }
        return 0.0;
    };

    // Budgets are scaled with the same critical value at which the interval
    // will be judged in the common many-strata regime; the variance upper
    // bound absorbs the small-sample gap between this and the Student-t the
    // estimator reports with.
    const double z = normal_quantile(0.5 + 0.5 * target.confidence);
    const double rel = target.relative_error;
    // The variance bound's failure probability is one third of the
    // complementary confidence (0.95 -> ~0.017), so planning caution scales
    // with the requested confidence instead of adding a separate constant:
    // the budget's allowed failure splits evenly over the bound on each
    // moment and the interval itself.
    const double delta = (1.0 - target.confidence) / 3.0;

    std::vector<std::uint64_t> target_n(H, 0);
    for (std::size_t mid = 0; mid < k; ++mid) {
        const auto measure = static_cast<MeasureId>(mid);
        const double mu = estimate_for(AggregateOp::Avg, measure);

        std::vector<double> sh_sum(H), sh_count(H), sh_avg(H);
        for (std::size_t h = 0; h < H; ++h) {
            const Spread sp = spread_from_observed(strata[h].observed[mid], mu,
                                                   delta);
            sh_sum[h] = sp.sh_sum;
            sh_count[h] = sp.sh_count;
            sh_avg[h] = sp.sh_avg;
            // A stratum somehow still below two samples cannot be planned
            // from its own statistics; demand its pilot before anything else.
            const auto& obs = strata[h].observed[mid];
            if (obs.m < 2 && obs.N > 0) {
                const std::uint64_t m0 = cfg_.pilot_sample_size;
                target_n[h] = std::max(
                    target_n[h], obs.N <= 2 * m0 ? obs.N : m0);
            }
        }

        const double t_sum = std::max(
            std::abs(estimate_for(AggregateOp::Sum, measure)), tau_sum);
        const double t_count = std::max(
            std::abs(estimate_for(AggregateOp::CountMeasure, measure)),
            tau_count);
        const double v_sum =
            rel > 0.0 ? std::pow(rel * t_sum / z, 2.0) : 0.0;
        const double v_count =
            rel > 0.0 ? std::pow(rel * t_count / z, 2.0) : 0.0;

        const auto n_sum = neyman_alloc(N, sh_sum, v_sum);
        const auto n_count = neyman_alloc(N, sh_count, v_count);
        for (std::size_t h = 0; h < H; ++h) {
            target_n[h] = std::max({target_n[h], n_sum[h], n_count[h]});
        }

        // AVG budget: Var(mu_hat) = (1/T_count^2) sum_h N_h^2 (1-f_h)
        // s_e,h^2 / m_h, so allocating the residual spreads under the budget
        // (rel |mu| / z)^2 * T_count^2 bounds the AVG interval directly.
        const double t_count_raw =
            estimate_for(AggregateOp::CountMeasure, measure);
        if (rel > 0.0 && std::isfinite(mu) && t_count_raw > 0.0) {
            const double half = rel * std::max(std::abs(mu), tau_avg) / z;
            const double v_avg = half * half * t_count_raw * t_count_raw;
            const auto n_avg = neyman_alloc(N, sh_avg, v_avg);
            for (std::size_t h = 0; h < H; ++h) {
                target_n[h] = std::max(target_n[h], n_avg[h]);
            }
        }
    }

    AllocationPlan plan;
    plan.target.resize(H);
    for (std::size_t h = 0; h < H; ++h) {
        std::uint64_t t = std::min(target_n[h], strata[h].N);
        t = std::max(t, strata[h].sampled);  // cumulative targets are monotone
        // Finish rule: a draw that reads most of what remains is better spent
        // reading the stratum whole -- nearly the same cost, zero residual
        // variance, and a persistent exact summary when the stratum persists.
        const std::uint64_t remaining = strata[h].N - strata[h].sampled;
        if (t > strata[h].sampled &&
            static_cast<double>(t - strata[h].sampled) >
                cfg_.exactification_sample_fraction *
                    static_cast<double>(remaining)) {
            t = strata[h].N;
        }
        plan.target[h] = t;
        plan.remaining_residual += remaining;
        if (t > strata[h].sampled) {
            plan.planned_new_reads += t - strata[h].sampled;
            plan.no_target_increase = false;
        }
    }
    return plan;
}

}  // namespace a3i

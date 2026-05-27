#include "a3i/aqp/estimator.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "a3i/util/tdist.hpp"

namespace a3i {

namespace {

constexpr double kMagnitudeFloorDelta = 1e-6;
// Fixed 95% normal deviate for the Wilson center used at sample proportions
// of exactly zero or one, so a small all-present/all-missing sample is not
// treated as deterministic.
constexpr double kWilsonZ = 1.959963984540054;

// Per-stratum variance components retained for the Satterthwaite degrees of
// freedom and for the linearized AVG variance.
struct Component {
    double        var_sum = 0.0;
    double        var_count = 0.0;
    double        cov = 0.0;
    std::uint64_t m = 0;
};

double satterthwaite_df(double total_var,
                        const std::vector<double>& comp_var,
                        const std::vector<std::uint64_t>& comp_m) {
    double denom = 0.0;
    for (std::size_t i = 0; i < comp_var.size(); ++i) {
        const double dfh =
            comp_m[i] > 1 ? static_cast<double>(comp_m[i] - 1) : 1.0;
        denom += comp_var[i] * comp_var[i] / dfh;
    }
    if (denom <= 0.0) return std::numeric_limits<double>::infinity();
    return total_var * total_var / denom;
}

AggregateEstimate certified(AggregateOp op, MeasureId mid, double value,
                            double total_var,
                            const std::vector<double>& comp_var,
                            const std::vector<std::uint64_t>& comp_m,
                            double tau, bool estimable, double confidence) {
    AggregateEstimate e;
    e.op = op;
    e.measure_id = mid;
    e.estimate = value;

    if (!estimable) {
        // A stratum too small to estimate its variance leaves the interval
        // uncertifiable; report an unbounded relative half-width so the
        // stopping rule keeps sampling.
        e.ci_low = -std::numeric_limits<double>::infinity();
        e.ci_high = std::numeric_limits<double>::infinity();
        e.relative_half_width = std::numeric_limits<double>::infinity();
        e.effective_df = 0.0;
        e.exact = false;
        return e;
    }

    if (total_var <= 0.0) {
        // No residual variance: every contributor was exact.
        e.ci_low = value;
        e.ci_high = value;
        e.relative_half_width = 0.0;
        e.effective_df = 0.0;
        e.exact = true;
        return e;
    }

    const double nu = satterthwaite_df(total_var, comp_var, comp_m);
    const double crit = t_quantile(0.5 + 0.5 * confidence, nu);
    const double margin = crit * std::sqrt(total_var);
    e.ci_low = value - margin;
    e.ci_high = value + margin;
    const double denom = std::max(std::abs(value), tau);
    e.relative_half_width = denom > 0.0 ? margin / denom
                                        : std::numeric_limits<double>::infinity();
    e.effective_df = nu;
    e.exact = false;
    return e;
}

}  // namespace

std::vector<AggregateEstimate> Estimator::estimate(
    const ExactBucket& exact_bucket, std::uint64_t total_count,
    const std::vector<std::vector<StratumSample>>& strata_by_measure,
    double global_mean_abs, double confidence) const {
    const std::size_t k = strata_by_measure.size();
    std::vector<AggregateEstimate> out;
    out.reserve(3 * k + 1);

    for (MeasureId mid = 0; mid < k; ++mid) {
        const auto& strata = strata_by_measure[mid];

        double t_sum =
            mid < exact_bucket.sum_by_measure.size()
                ? exact_bucket.sum_by_measure[mid]
                : 0.0;
        double t_count =
            mid < exact_bucket.count_by_measure.size()
                ? static_cast<double>(exact_bucket.count_by_measure[mid])
                : 0.0;
        double var_sum = 0.0, var_count = 0.0, cov = 0.0;
        std::uint64_t sum_population = 0;
        std::vector<Component> comps;
        bool estimable = true;

        for (const StratumSample& s : strata) {
            if (s.N == 0) continue;
            sum_population += s.N;

            if (s.m >= s.N) {  // fully read -> exact, zero variance
                t_sum += s.S;
                t_count += static_cast<double>(s.n);
                continue;
            }
            if (s.m < 2) {  // cannot estimate this stratum's variance
                if (s.m > 0) {
                    const double scale = static_cast<double>(s.N) /
                                         static_cast<double>(s.m);
                    t_sum += scale * s.S;
                    t_count += scale * static_cast<double>(s.n);
                }
                estimable = false;
                continue;
            }

            const double N = static_cast<double>(s.N);
            const double m = static_cast<double>(s.m);
            const double n = static_cast<double>(s.n);
            const double fpc = 1.0 - m / N;

            double s2z = (s.Q - s.S * s.S / m) / (m - 1.0);
            if (s2z < 0.0) s2z = 0.0;
            const double s_hat = N * s.S / m;
            double v_sum_h = N * N * (s2z / m) * fpc;
            if (v_sum_h < 0.0) v_sum_h = 0.0;

            const double p_hat = n / m;
            const double p_adj =
                (p_hat <= 0.0 || p_hat >= 1.0)
                    ? (n + kWilsonZ * kWilsonZ / 2.0) / (m + kWilsonZ * kWilsonZ)
                    : p_hat;
            const double c_hat = N * p_hat;
            double v_cnt_h = N * N * (p_adj * (1.0 - p_adj) / (m - 1.0)) * fpc;
            if (v_cnt_h < 0.0) v_cnt_h = 0.0;

            const double cov_h =
                N * N * (s.S * (m - n) / (m * m * (m - 1.0))) * fpc;

            t_sum += s_hat;
            t_count += c_hat;
            var_sum += v_sum_h;
            var_count += v_cnt_h;
            cov += cov_h;
            comps.push_back({v_sum_h, v_cnt_h, cov_h, s.m});
        }

        const double tau_sum =
            kMagnitudeFloorDelta * global_mean_abs *
            static_cast<double>(sum_population);
        const double tau_count =
            std::max(1.0, kMagnitudeFloorDelta *
                              static_cast<double>(sum_population));
        const double tau_avg = kMagnitudeFloorDelta * global_mean_abs;

        // SUM
        std::vector<double> cv_sum;
        std::vector<std::uint64_t> cm;
        cv_sum.reserve(comps.size());
        cm.reserve(comps.size());
        for (const Component& c : comps) {
            cv_sum.push_back(c.var_sum);
            cm.push_back(c.m);
        }
        out.push_back(certified(AggregateOp::Sum, mid, t_sum, var_sum, cv_sum,
                                cm, tau_sum, estimable, confidence));

        // COUNT(measure)
        std::vector<double> cv_cnt;
        cv_cnt.reserve(comps.size());
        for (const Component& c : comps) cv_cnt.push_back(c.var_count);
        out.push_back(certified(AggregateOp::CountMeasure, mid, t_count,
                                var_count, cv_cnt, cm, tau_count, estimable,
                                confidence));

        // AVG (ratio, delta method linearized at the combined mean)
        AggregateEstimate avg;
        avg.op = AggregateOp::Avg;
        avg.measure_id = mid;
        if (t_count > 0.0) {
            const double mu = t_sum / t_count;
            const double inv_tc2 = 1.0 / (t_count * t_count);
            double var_mu =
                inv_tc2 * (var_sum - 2.0 * mu * cov + mu * mu * var_count);
            if (var_mu < 0.0) var_mu = 0.0;
            std::vector<double> cv_avg;
            cv_avg.reserve(comps.size());
            for (const Component& c : comps) {
                double vh = inv_tc2 * (c.var_sum - 2.0 * mu * c.cov +
                                       mu * mu * c.var_count);
                if (vh < 0.0) vh = 0.0;
                cv_avg.push_back(vh);
            }
            avg = certified(AggregateOp::Avg, mid, mu, var_mu, cv_avg, cm,
                            tau_avg, estimable, confidence);
        } else {
            // No qualifying non-missing rows: AVG is undefined but exact.
            avg.estimate = std::numeric_limits<double>::quiet_NaN();
            avg.ci_low = avg.estimate;
            avg.ci_high = avg.estimate;
            avg.relative_half_width = 0.0;
            avg.effective_df = 0.0;
            avg.exact = true;
        }
        out.push_back(avg);
    }

    AggregateEstimate count_star;
    count_star.op = AggregateOp::CountStar;
    count_star.measure_id = 0;
    count_star.estimate = static_cast<double>(total_count);
    count_star.ci_low = count_star.estimate;
    count_star.ci_high = count_star.estimate;
    count_star.relative_half_width = 0.0;
    count_star.effective_df = 0.0;
    count_star.exact = true;
    out.push_back(count_star);

    return out;
}

}  // namespace a3i

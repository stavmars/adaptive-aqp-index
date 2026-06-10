#include "a3i/util/tdist.hpp"

#include <cmath>
#include <limits>

namespace a3i {

namespace {

constexpr double kPi = 3.14159265358979311599796346854;

}  // namespace

// Acklam's rational approximation to the inverse standard-normal CDF,
// refined by one Halley step so the result is accurate to roughly machine
// precision across the whole open interval.
double normal_quantile(double p) {
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();

    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                               -2.759285104469687e+02, 1.383577518672690e+02,
                               -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                               -1.556989798598866e+02, 6.680131188771972e+01,
                               -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                               -2.400758277161838e+00, -2.549732539343734e+00,
                               4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                               2.445134137142996e+00, 3.754408661907416e+00};

    const double plow = 0.02425;
    const double phigh = 1.0 - plow;
    double x;
    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= phigh) {
        const double q = p - 0.5;
        const double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
            q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    // One Halley refinement step against the true CDF.
    const double e = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
    const double u = e * std::sqrt(2.0 * kPi) * std::exp(x * x / 2.0);
    x = x - u / (1.0 + x * u / 2.0);
    return x;
}

// Hill's algorithm for Student's t-quantiles (Hill 1970): an asymptotic
// series in 1/nu with exact closed forms for one and two degrees of freedom.
// `two_tail` is the total tail area P(|T| > t); the returned value is the
// positive t cutting off that area.
static double studentt_two_tail(double two_tail, double nu) {
    if (nu == 1.0) {
        const double arg = two_tail * kPi / 2.0;
        return std::cos(arg) / std::sin(arg);
    }
    if (nu == 2.0) {
        return std::sqrt(2.0 / (two_tail * (2.0 - two_tail)) - 2.0);
    }

    const double a = 1.0 / (nu - 0.5);
    const double b = 48.0 / (a * a);
    double c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36;
    const double d =
        ((94.5 / (b + c) - 3.0) / b + 1.0) * std::sqrt(a * kPi / 2.0) * nu;
    double y = std::pow(d * two_tail, 2.0 / nu);

    if (y > 0.05 + a) {
        // Large-deviate branch: start from the normal quantile and correct.
        const double x = normal_quantile(1.0 - two_tail / 2.0);
        y = x * x;
        if (nu < 5.0) c = c + 0.3 * (nu - 4.5) * (x + 0.6);
        c = (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b + c;
        y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b +
             1.0) *
            x;
        y = a * y * y;
        y = (y > 0.002) ? std::exp(y) - 1.0 : 0.5 * y * y + y;
    } else {
        y = ((1.0 / (((nu + 6.0) / (nu * y) - 0.089 * d - 0.822) * (nu + 2.0) *
                     3.0) +
              0.5 / (nu + 4.0)) *
                 y -
             1.0) *
                (nu + 1.0) / (nu + 2.0) +
            1.0 / y;
    }
    return std::sqrt(nu * y);
}

double t_quantile(double p, double nu) {
    if (nu < 1.0) nu = 1.0;
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();
    if (p == 0.5) return 0.0;

    // Very large degrees of freedom collapse to the normal quantile, where
    // Hill's series loses accuracy in the deep tails.
    if (nu > 1.0e7) return normal_quantile(p);

    const bool upper = p > 0.5;
    const double tail_prob = upper ? (1.0 - p) : p;  // one-sided tail area
    const double t = studentt_two_tail(2.0 * tail_prob, nu);
    return upper ? t : -t;
}

// Wilson-Hilferty transform: the cube root of a chi-squared variable over its
// degrees of freedom is approximately normal with mean 1 - 2/(9 nu) and
// variance 2/(9 nu), so the quantile is the back-transformed normal quantile.
double chi_squared_quantile(double p, double nu) {
    if (nu < 1.0) nu = 1.0;
    if (p <= 0.0) p = std::numeric_limits<double>::min();
    if (p >= 1.0) p = 1.0 - 1e-16;

    const double z = normal_quantile(p);
    const double s = 2.0 / (9.0 * nu);
    const double cube = 1.0 - s + z * std::sqrt(s);
    const double x = nu * cube * cube * cube;
    // The cube can go non-positive in the far lower tail at tiny nu; the
    // distribution itself is strictly positive, so floor accordingly.
    return x > 0.0 ? x : std::numeric_limits<double>::min();
}

}  // namespace a3i

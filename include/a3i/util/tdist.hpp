// Student-t and standard-normal quantiles, dependency-free.
//
// The confidence interval reported for an aggregate uses a Student-t
// critical value whose degrees of freedom come from a Welch-Satterthwaite
// approximation, so the degrees of freedom are generally fractional and the
// quantile must be evaluated, not looked up. Both routines are closed-form
// rational/series approximations accurate to well within the precision the
// stopping rule needs; they allocate nothing and pull in no external math
// library beyond <cmath>.

#pragma once

namespace a3i {

/// Inverse standard-normal CDF: the value z with P(Z <= z) = p, for
/// 0 < p < 1. Returns -inf / +inf at the open boundaries.
double normal_quantile(double p);

/// Inverse Student-t CDF: the value t with P(T <= t) = p for a
/// distribution with `nu` degrees of freedom (nu >= 1, may be fractional).
/// Symmetric about zero; `nu` is clamped to at least 1.
double t_quantile(double p, double nu);

}  // namespace a3i

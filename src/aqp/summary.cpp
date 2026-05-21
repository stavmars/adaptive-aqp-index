#include "a3i/aqp/summary.hpp"

#include <cmath>
#include <stdexcept>

namespace a3i {

void MomentStats::add_if_present(double x) {
    if (std::isnan(x)) return;
    ++non_nan_count;
    const double delta = x - mean;
    mean += delta / static_cast<double>(non_nan_count);
    const double delta2 = x - mean;
    m2 += delta * delta2;
}

double MomentStats::sample_variance() const {
    if (non_nan_count < 2) return 0.0;
    const double v = m2 / static_cast<double>(non_nan_count - 1);
    return v < 0.0 ? 0.0 : v;
}

void MomentStats::merge(const MomentStats& other) {
    if (other.non_nan_count == 0) return;
    if (non_nan_count == 0) {
        *this = other;
        return;
    }
    const double na = static_cast<double>(non_nan_count);
    const double nb = static_cast<double>(other.non_nan_count);
    const double n  = na + nb;
    const double delta = other.mean - mean;
    mean += delta * nb / n;
    m2   += other.m2 + delta * delta * na * nb / n;
    non_nan_count += other.non_nan_count;
}

SampleTracker::SampleTracker(std::uint64_t stratum_size)
    : bits_((stratum_size + 63) / 64, 0), size_(stratum_size) {}

bool SampleTracker::contains(IndexPos local) const {
    const std::uint64_t i = local;
    if (i >= size_) return false;
    return (bits_[i >> 6] >> (i & 63)) & 1ULL;
}

void SampleTracker::add(IndexPos local) {
    const std::uint64_t i = local;
    if (i >= size_) {
        throw std::out_of_range("SampleTracker::add position out of range");
    }
    std::uint64_t& word = bits_[i >> 6];
    const std::uint64_t mask = 1ULL << (i & 63);
    if (!(word & mask)) {
        word |= mask;
        ++count_;
    }
}

}  // namespace a3i

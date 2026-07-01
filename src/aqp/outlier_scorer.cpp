#include "a3i/aqp/outlier_scorer.hpp"

#include <algorithm>
#include <cmath>

namespace a3i {

namespace {
// Min-heap on score: the front is the smallest retained score, so a new row
// displaces it only when it scores higher.
bool score_greater(const OutlierScorer::Entry& a, const OutlierScorer::Entry& b) {
    return a.score > b.score;
}
}  // namespace

OutlierScorer::OutlierScorer(std::vector<double> mean,
                             std::vector<double> inv_scale,
                             std::uint64_t budget)
    : mean_(std::move(mean)),
      inv_scale_(std::move(inv_scale)),
      budget_(budget) {
    heap_.reserve(static_cast<std::size_t>(
        std::min<std::uint64_t>(budget_, std::uint64_t{1} << 20)));
}

double OutlierScorer::score_of(std::span<const double> values) const {
    // The largest mean-normalized squared deviation over the measures:
    // max_m ((x_m - mean_m) / mean_m)^2, a per-measure contribution to
    // CV^2 = (sigma/mu)^2 (relative variance). The budget-many highest scores
    // are the rows whose removal most reduces the sampled population's relative
    // standard deviation, to which the SUM sampling error is proportional.
    // Two-sided; ignores missing values and measures with inv_scale 0.
    double best = 0.0;
    const std::size_t k = std::min(values.size(), inv_scale_.size());
    for (std::size_t m = 0; m < k; ++m) {
        const double inv = inv_scale_[m];  // 1/mean, 0 skips the measure
        if (inv == 0.0) continue;  // only a zero scale skips; the score squares inv
        const double x = values[m];
        if (std::isnan(x)) continue;
        const double d = (x - mean_[m]) * inv;  // (x - mean) / mean
        const double s = d * d;
        if (s > best) best = s;
    }
    return best;
}

void OutlierScorer::observe(RowId row, std::span<const double> values) {
    if (budget_ == 0) return;
    const double s = score_of(values);
    if (!(s > 0.0)) return;  // a row at the mean for every measure is not extreme
    if (heap_.size() < budget_) {
        heap_.push_back({s, row});
        std::push_heap(heap_.begin(), heap_.end(), score_greater);
    } else if (s > heap_.front().score) {
        std::pop_heap(heap_.begin(), heap_.end(), score_greater);
        heap_.back() = {s, row};
        std::push_heap(heap_.begin(), heap_.end(), score_greater);
    }
}

std::vector<RowId> OutlierScorer::finalize() {
    std::vector<RowId> out;
    out.reserve(heap_.size());
    for (const Entry& e : heap_) out.push_back(e.row);
    heap_.clear();
    heap_.shrink_to_fit();
    return out;
}

}  // namespace a3i

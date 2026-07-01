// Selects the budget-many rows that most reduce the sampled population's
// (relative) variance: holding these rows out of sampling and contributing them
// exactly leaves the rest with a smaller standard deviation, hence a smaller
// sampling error for the same sample size.
//
// A row's score is the largest mean-normalized squared deviation over the
// measures, max_m ((x_m - mean_m) / mean_m)^2 -- each term is that row's
// contribution to the measure's CV^2 = (sigma/mu)^2 (its relative variance), so
// the highest-scoring rows are those whose removal most shrinks the relative
// standard deviation the SUM sampling error is proportional to. Two-sided;
// ignores missing values; a measure with 1/mean == 0 is skipped (e.g. a zero
// mean). Rows are streamed past observe() while a bounded min-heap keeps the
// running top-budget set; finalize() returns their row ids.
//
// The mean and 1/mean are exact per-measure population statistics supplied by
// the caller, so no running estimate is needed.

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "a3i/core/types.hpp"

namespace a3i {

class OutlierScorer {
public:
    /// `mean[m]` and `inv_scale[m]` are the per-measure mean and 1/mean; pass
    /// inv_scale[m] == 0 for a measure that should never contribute (e.g. a
    /// zero mean, which has no mean-normalized scale). The score squares 1/mean,
    /// so a negative mean is fine. `budget` is the number of rows to retain.
    OutlierScorer(std::vector<double> mean, std::vector<double> inv_scale,
                  std::uint64_t budget);

    /// Fold one row's measure values into the running top-budget set.
    void observe(RowId row, std::span<const double> values);

    /// Row ids of the retained top-budget rows (unordered); empties the heap.
    std::vector<RowId> finalize();

    struct Entry {
        double score;
        RowId  row;
    };

private:
    double score_of(std::span<const double> values) const;

    std::vector<double> mean_;
    std::vector<double> inv_scale_;  // 1/scale per measure; 0 => measure ignored
    std::uint64_t       budget_ = 0;
    std::vector<Entry>  heap_;        // min-heap on score, capacity budget_
};

}  // namespace a3i

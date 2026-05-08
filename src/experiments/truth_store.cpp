#include "a3i/experiments/truth_store.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace a3i {

namespace {

AggregateEstimate exact_estimate(AggregateOp op, MeasureId measure_id, double value) {
    AggregateEstimate e;
    e.op                  = op;
    e.measure_id          = measure_id;
    e.estimate            = value;
    e.ci_low              = value;
    e.ci_high             = value;
    e.relative_half_width = 0.0;
    e.effective_df        = 0.0;
    e.exact               = true;
    return e;
}

}  // namespace

QueryResult exact_scan(const BinaryColumnStore& store,
                       const DatasetSchema&     schema,
                       const RangeQuery&        query) {
    const std::size_t d = schema.dimension_names.size();
    const std::size_t k = schema.measure_names.size();
    if (query.predicate.dims.size() != d) {
        throw std::invalid_argument(
            "exact_scan: query predicate dimensionality does not match schema");
    }

    const std::size_t n = store.row_count();

    // Eager dimension columns; the predicate is tested coordinate by
    // coordinate per row without touching any measure.
    std::vector<std::span<const double>> dim_columns;
    dim_columns.reserve(d);
    for (std::size_t i = 0; i < d; ++i) {
        dim_columns.push_back(
            store.dimension_column(static_cast<DimensionId>(i)));
    }

    std::vector<double>        sums(k, 0.0);          // null-as-zero SUM
    std::vector<std::uint64_t> non_nan_counts(k, 0);  // COUNT(measure)
    std::uint64_t              qualifying = 0;        // COUNT(*)

    std::vector<double> point(d, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t i = 0; i < d; ++i) point[i] = dim_columns[i][r];
        if (!query.predicate.contains_point(point)) continue;
        ++qualifying;
        for (std::size_t m = 0; m < k; ++m) {
            const double v =
                store.measure_value(static_cast<RowId>(r), static_cast<MeasureId>(m));
            if (!std::isnan(v)) {
                sums[m] += v;
                ++non_nan_counts[m];
            }
        }
    }

    QueryResult result;
    result.aggregates.reserve(3 * k + 1);
    for (std::size_t m = 0; m < k; ++m) {
        const auto mid   = static_cast<MeasureId>(m);
        const auto count = non_nan_counts[m];
        const double avg =
            count == 0 ? std::numeric_limits<double>::quiet_NaN() : sums[m] / static_cast<double>(count);
        result.aggregates.push_back(exact_estimate(AggregateOp::Sum, mid, sums[m]));
        result.aggregates.push_back(
            exact_estimate(AggregateOp::CountMeasure, mid, static_cast<double>(count)));
        result.aggregates.push_back(exact_estimate(AggregateOp::Avg, mid, avg));
    }
    result.aggregates.push_back(
        exact_estimate(AggregateOp::CountStar, 0, static_cast<double>(qualifying)));

    auto& metrics            = result.metrics;
    metrics.method           = "exact_scan";
    metrics.substrate        = "none";
    metrics.status           = "exact";
    metrics.stop_reason      = "exact";
    metrics.rows_examined    = static_cast<std::uint64_t>(n);
    metrics.measure_reads    = qualifying * static_cast<std::uint64_t>(k);
    metrics.target_satisfied = true;
    return result;
}

}  // namespace a3i

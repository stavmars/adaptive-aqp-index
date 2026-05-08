// Exact range-aggregate ground truth (the differential oracle).
//
// `exact_scan` answers a range query by a full pass over the binary
// columns: it tests every row's dimensions against the predicate and, for
// the qualifying rows, reads each measure and folds it into exact SUM /
// COUNT(measure) / AVG accumulators, plus the measure-free COUNT(*). It
// uses no access path and no cached summaries, so it is the independent
// reference every engine-produced answer is checked against.
//
// Measure semantics: NaN means missing. SUM and COUNT(measure) range over
// the non-NaN values (SUM is therefore null-as-zero); AVG = SUM/COUNT, NaN
// when the qualifying non-NaN count is zero. COUNT(*) is the qualifying row
// count and needs no measure access.

#pragma once

#include "a3i/core/query.hpp"
#include "a3i/core/schema.hpp"
#include "a3i/storage/binary_column_store.hpp"

namespace a3i {

/// Compute exact aggregates for `query` over the rows in `store`, scoped to
/// the dimensions and measures named by `schema`. Returns one estimate per
/// (measure x {SUM, COUNT(measure), AVG}) followed by a single COUNT(*); all
/// estimates are exact (zero-width intervals). Throws std::invalid_argument
/// if the predicate's dimensionality does not match the schema.
QueryResult exact_scan(const BinaryColumnStore& store,
                       const DatasetSchema&     schema,
                       const RangeQuery&        query);

}  // namespace a3i

// Per-index dataset description.
//
// A DatasetSchema captures everything one index instance needs to know
// about its underlying data: which dimension columns it organizes,
// which measure columns it serves, the domain bounds those dimensions
// live in, the row count, and a path to the binary manifest that backs
// the columns. The schema is fixed for the lifetime of the index.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "a3i/core/geometry.hpp"

namespace a3i {

struct DatasetSchema {
    /// Dimension columns, in dimension order. DimensionId == position here.
    std::vector<std::string> dimension_names;

    /// Measure columns this index serves. MeasureId == position here.
    std::vector<std::string> measure_names;

    /// Per-dimension [low, high) domain bounds, aligned with dimension_names.
    HyperRect domain_bounds;

    /// Total number of objects (rows) in the dataset.
    std::uint64_t object_count = 0;

    /// Path to the manifest describing the on-disk binary columns.
    /// Per-column file paths inside the manifest are resolved relative
    /// to the manifest's own directory, so a prepared directory is
    /// relocatable.
    std::string binary_manifest_path;
};

}  // namespace a3i

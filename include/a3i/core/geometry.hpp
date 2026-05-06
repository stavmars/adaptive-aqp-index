// Geometric primitives used throughout the index and query layer.
//
// The half-open convention [low, high) is global: a point x satisfies a
// range iff low <= x < high. Predicates over these types are added when
// the query and access-path layers need them.

#pragma once

#include <vector>

namespace a3i {

struct Range {
    double low  = 0.0;
    double high = 0.0;
};

struct HyperRect {
    std::vector<Range> dims;
};

}  // namespace a3i

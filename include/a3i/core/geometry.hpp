// Geometric primitives used throughout the index and query layer.
//
// The half-open convention [low, high) is global: a point x satisfies a
// range iff low <= x < high. The same convention governs every predicate
// below, so a rectangle that ends exactly where another begins does not
// overlap it.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace a3i {

struct Range {
    double low  = 0.0;
    double high = 0.0;
};

struct HyperRect {
    std::vector<Range> dims;

    /// True iff `p` lies inside the rectangle under the half-open rule
    /// (`low <= p[i] < high` on every axis). `p` must have one coordinate
    /// per dimension.
    bool contains_point(std::span<const double> p) const {
        if (p.size() != dims.size()) return false;
        for (std::size_t i = 0; i < dims.size(); ++i) {
            if (p[i] < dims[i].low || p[i] >= dims[i].high) return false;
        }
        return true;
    }

    /// True iff every point of `o` is also a point of `*this`, i.e. on each
    /// axis `low <= o.low` and `o.high <= high`. Rectangles must share the
    /// same dimensionality.
    bool contains_rect(const HyperRect& o) const {
        if (o.dims.size() != dims.size()) return false;
        for (std::size_t i = 0; i < dims.size(); ++i) {
            if (o.dims[i].low < dims[i].low || o.dims[i].high > dims[i].high) {
                return false;
            }
        }
        return true;
    }

    /// True iff the two rectangles share at least one point. Under the
    /// half-open rule axes overlap when `low < o.high && o.low < high`, so
    /// abutting rectangles (`[0,1)` and `[1,2)`) do not intersect.
    bool intersects(const HyperRect& o) const {
        if (o.dims.size() != dims.size()) return false;
        for (std::size_t i = 0; i < dims.size(); ++i) {
            if (dims[i].low >= o.dims[i].high || o.dims[i].low >= dims[i].high) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace a3i

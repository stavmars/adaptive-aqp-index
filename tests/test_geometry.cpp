// Tests for the half-open geometry predicates.

#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "a3i/core/geometry.hpp"

namespace {

a3i::HyperRect rect(std::vector<a3i::Range> dims) {
    return a3i::HyperRect{std::move(dims)};
}

bool contains(const a3i::HyperRect& r, std::initializer_list<double> p) {
    std::vector<double> v(p);
    return r.contains_point(std::span<const double>(v.data(), v.size()));
}

}  // namespace

TEST(Geometry, ContainsPointHalfOpen) {
    auto r = rect({{0.0, 1.0}, {0.0, 1.0}});
    EXPECT_TRUE(contains(r, {0.0, 0.0}));   // low edge included
    EXPECT_TRUE(contains(r, {0.5, 0.5}));
    EXPECT_FALSE(contains(r, {1.0, 0.5}));  // high edge excluded
    EXPECT_FALSE(contains(r, {0.5, 1.0}));
    EXPECT_FALSE(contains(r, {-0.001, 0.5}));
}

TEST(Geometry, ContainsPointRejectsWrongArity) {
    auto r = rect({{0.0, 1.0}, {0.0, 1.0}});
    EXPECT_FALSE(contains(r, {0.5}));
    EXPECT_FALSE(contains(r, {0.5, 0.5, 0.5}));
}

TEST(Geometry, ContainsRect) {
    auto outer = rect({{0.0, 10.0}, {0.0, 10.0}});
    EXPECT_TRUE(outer.contains_rect(rect({{1.0, 9.0}, {1.0, 9.0}})));
    EXPECT_TRUE(outer.contains_rect(outer));  // a rect contains itself
    EXPECT_FALSE(outer.contains_rect(rect({{-1.0, 5.0}, {1.0, 9.0}})));
    EXPECT_FALSE(outer.contains_rect(rect({{1.0, 5.0}, {1.0, 10.1}})));
}

TEST(Geometry, IntersectsAndAbutting) {
    auto a = rect({{0.0, 1.0}, {0.0, 1.0}});
    EXPECT_TRUE(a.intersects(rect({{0.5, 1.5}, {0.5, 1.5}})));   // overlap
    EXPECT_TRUE(a.intersects(a));                                // self
    EXPECT_FALSE(a.intersects(rect({{1.0, 2.0}, {0.0, 1.0}})));  // abut on x
    EXPECT_FALSE(a.intersects(rect({{0.0, 1.0}, {1.0, 2.0}})));  // abut on y
    EXPECT_FALSE(a.intersects(rect({{2.0, 3.0}, {2.0, 3.0}})));  // disjoint
}

TEST(Geometry, MismatchedArityNeverIntersectsOrContains) {
    auto a = rect({{0.0, 1.0}, {0.0, 1.0}});
    auto b = rect({{0.0, 1.0}});
    EXPECT_FALSE(a.intersects(b));
    EXPECT_FALSE(a.contains_rect(b));
}

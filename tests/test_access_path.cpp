#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "a3i/access_path/substrate_factory.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/adaptive_kd_access_path.hpp"

namespace {

using namespace a3i;

// A small 2D fixture: nine points spread across the [0,10) x [0,10) domain.
IndexTable make_table() {
    std::vector<double> xs = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> ys = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    return IndexTable::from_columns({xs, ys});
}

HyperRect domain() { return HyperRect{{{0.0, 10.0}, {0.0, 10.0}}}; }

SubstrateConfig config() {
    SubstrateConfig cfg;
    cfg.domain_bounds = domain();
    return cfg;
}

// Assert the partition invariants over the substrate's active partitions:
// dense coverage of [0, N) with no overlap and only active leaves.
void check_partition_invariants(const AdaptiveAccessPath& path, std::size_t n) {
    auto active = path.active_partitions();
    ASSERT_FALSE(active.empty());

    std::vector<bool> covered(n, false);
    for (PartitionId id : active) {
        PartitionView pv = path.partition(id);
        EXPECT_TRUE(pv.active);
        EXPECT_LE(pv.begin, pv.end);
        EXPECT_LE(pv.end, static_cast<IndexPos>(n));
        for (IndexPos p = pv.begin; p < pv.end; ++p) {
            ASSERT_FALSE(covered[p]) << "position " << p << " in two partitions";
            covered[p] = true;
        }
    }
    for (std::size_t p = 0; p < n; ++p) {
        EXPECT_TRUE(covered[p]) << "position " << p << " in no active partition";
    }
}

TEST(AdaptiveKdAccessPath, EnsureBuiltCreatesSingleRoot) {
    IndexTable table = make_table();
    AdaptiveKdAccessPath path(config());
    path.prepare(table);
    path.ensure_built();

    auto active = path.active_partitions();
    ASSERT_EQ(active.size(), 1u);
    EXPECT_EQ(active[0], 0u);

    PartitionView root = path.partition(0);
    EXPECT_EQ(root.id, 0u);
    EXPECT_EQ(root.begin, 0u);
    EXPECT_EQ(root.end, static_cast<IndexPos>(table.size()));
    EXPECT_TRUE(root.active);
    EXPECT_EQ(root.bounds.dims.size(), 2u);
}

TEST(AdaptiveKdAccessPath, EnsureBuiltIsIdempotent) {
    IndexTable table = make_table();
    AdaptiveKdAccessPath path(config());
    path.prepare(table);
    path.ensure_built();
    path.ensure_built();
    EXPECT_EQ(path.active_partitions().size(), 1u);
}

TEST(AdaptiveKdAccessPath, LocateFullyContained) {
    IndexTable table = make_table();
    AdaptiveKdAccessPath path(config());
    path.prepare(table);
    path.ensure_built();

    // A query that covers the whole domain contains the root.
    QueryPartitionSet set = path.locate(HyperRect{{{-1.0, 11.0}, {-1.0, 11.0}}});
    ASSERT_EQ(set.fully_contained.size(), 1u);
    EXPECT_EQ(set.fully_contained[0], 0u);
    EXPECT_TRUE(set.partial.empty());
}

TEST(AdaptiveKdAccessPath, LocatePartialOverlap) {
    IndexTable table = make_table();
    AdaptiveKdAccessPath path(config());
    path.prepare(table);
    path.ensure_built();

    // Overlaps the domain but does not contain it.
    QueryPartitionSet set = path.locate(HyperRect{{{2.0, 6.0}, {2.0, 6.0}}});
    EXPECT_TRUE(set.fully_contained.empty());
    ASSERT_EQ(set.partial.size(), 1u);
    EXPECT_EQ(set.partial[0], 0u);
}

TEST(AdaptiveKdAccessPath, LocateDisjoint) {
    IndexTable table = make_table();
    AdaptiveKdAccessPath path(config());
    path.prepare(table);
    path.ensure_built();

    // Entirely outside the domain (abutting at the upper edge counts as
    // disjoint under the half-open rule).
    QueryPartitionSet set = path.locate(HyperRect{{{10.0, 20.0}, {10.0, 20.0}}});
    EXPECT_TRUE(set.fully_contained.empty());
    EXPECT_TRUE(set.partial.empty());
}

TEST(AdaptiveKdAccessPath, RootHasNoParent) {
    IndexTable table = make_table();
    AdaptiveKdAccessPath path(config());
    path.prepare(table);
    path.ensure_built();
    EXPECT_FALSE(path.parent(0).has_value());
}

TEST(AdaptiveKdAccessPath, RefineRetiresNothing) {
    IndexTable table = make_table();
    AdaptiveKdAccessPath path(config());
    path.prepare(table);
    path.ensure_built();
    auto retired = path.refine(HyperRect{{{2.0, 6.0}, {2.0, 6.0}}}, table);
    EXPECT_TRUE(retired.empty());
    EXPECT_EQ(path.active_partitions().size(), 1u);
}

TEST(AdaptiveKdAccessPath, RangesNotRowIdOrdered) {
    AdaptiveKdAccessPath path(config());
    EXPECT_FALSE(path.ranges_are_row_id_ordered());
}

TEST(AdaptiveKdAccessPath, PartitionInvariantsHold) {
    IndexTable table = make_table();
    AdaptiveKdAccessPath path(config());
    path.prepare(table);
    path.ensure_built();
    check_partition_invariants(path, table.size());
}

TEST(AdaptiveKdAccessPath, EnsureBuiltBeforePrepareThrows) {
    AdaptiveKdAccessPath path(config());
    EXPECT_THROW(path.ensure_built(), std::logic_error);
}

TEST(AdaptiveKdAccessPath, UnknownPartitionThrows) {
    IndexTable table = make_table();
    AdaptiveKdAccessPath path(config());
    path.prepare(table);
    path.ensure_built();
    EXPECT_THROW(path.partition(99), std::out_of_range);
    EXPECT_THROW(path.parent(99), std::out_of_range);
}

}  // namespace

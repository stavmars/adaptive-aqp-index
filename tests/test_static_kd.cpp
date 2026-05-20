// Build correctness, partition invariants, locate classification, and the
// no-op refine for the static (fully-built) KD substrate.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "a3i/core/geometry.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/static_kd_access_path.hpp"

namespace {

using namespace a3i;

// A 16 x 16 integer grid of points over [0, 16) x [0, 16): 256 points,
// row id == (x * 16 + y) in load order.
IndexTable make_grid() {
    std::vector<double> xs;
    std::vector<double> ys;
    for (int x = 0; x < 16; ++x) {
        for (int y = 0; y < 16; ++y) {
            xs.push_back(static_cast<double>(x));
            ys.push_back(static_cast<double>(y));
        }
    }
    return IndexTable::from_columns({xs, ys});
}

SubstrateConfig config(std::uint32_t leaf_min_size) {
    SubstrateConfig cfg;
    cfg.domain_bounds   = HyperRect{{{0.0, 16.0}, {0.0, 16.0}}};
    cfg.leaf_min_size   = leaf_min_size;
    return cfg;
}

// Active partitions tile [0, n) disjointly, cover it, and have unique ids;
// their row_id payloads are a permutation of [0, n).
void check_cover(const StaticKdAccessPath& path, const IndexTable& table) {
    auto active = path.active_partitions();
    ASSERT_FALSE(active.empty());
    std::vector<bool> covered(table.size(), false);
    std::vector<PartitionId> ids;
    for (PartitionId id : active) {
        PartitionView pv = path.partition(id);
        EXPECT_TRUE(pv.active);
        EXPECT_LE(pv.begin, pv.end);
        EXPECT_LE(pv.end, static_cast<IndexPos>(table.size()));
        for (IndexPos p = pv.begin; p < pv.end; ++p) {
            ASSERT_FALSE(covered[p]) << "position double-covered";
            covered[p] = true;
        }
        ids.push_back(id);
    }
    for (std::size_t p = 0; p < table.size(); ++p) EXPECT_TRUE(covered[p]);
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end())
        << "duplicate partition id";

    std::vector<RowId> rows(table.row_ids().begin(), table.row_ids().end());
    std::sort(rows.begin(), rows.end());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        EXPECT_EQ(rows[i], static_cast<RowId>(i)) << "row id payload corrupted";
    }
}

// Every point physically inside a partition's range lies inside its bounds.
void check_points_in_bounds(const StaticKdAccessPath& path,
                            const IndexTable& table) {
    for (PartitionId id : path.active_partitions()) {
        PartitionView pv = path.partition(id);
        for (IndexPos p = pv.begin; p < pv.end; ++p) {
            EXPECT_TRUE(pv.bounds.contains_point(table.point(p)))
                << "point outside its partition bounds";
        }
    }
}

TEST(StaticKd, BuildSplitsDownToLeafSize) {
    IndexTable table = make_grid();
    StaticKdAccessPath path(config(/*leaf_min_size=*/16));
    path.prepare(table);
    path.ensure_built();

    auto active = path.active_partitions();
    EXPECT_GT(active.size(), 1u) << "a 256-point table must split";
    for (PartitionId id : active) {
        PartitionView pv = path.partition(id);
        EXPECT_LE(pv.end - pv.begin, 16u) << "leaf exceeds leaf_min_size";
    }
    check_cover(path, table);
    check_points_in_bounds(path, table);
}

TEST(StaticKd, SingleLeafWhenThresholdExceedsTable) {
    IndexTable table = make_grid();
    StaticKdAccessPath path(config(/*leaf_min_size=*/1000));
    path.prepare(table);
    path.ensure_built();
    EXPECT_EQ(path.active_partitions().size(), 1u);
    check_cover(path, table);
}

TEST(StaticKd, LocateClassifiesContainedAndPartial) {
    IndexTable table = make_grid();
    StaticKdAccessPath path(config(/*leaf_min_size=*/16));
    path.prepare(table);
    path.ensure_built();

    HyperRect q{{{3.0, 11.0}, {3.0, 11.0}}};
    QueryPartitionSet set = path.locate(q);

    // Fully-contained partitions are inside q and every one of their points
    // qualifies; partial partitions intersect q but are not inside it.
    for (PartitionId id : set.fully_contained) {
        PartitionView pv = path.partition(id);
        EXPECT_TRUE(q.contains_rect(pv.bounds));
        for (IndexPos p = pv.begin; p < pv.end; ++p) {
            EXPECT_TRUE(q.contains_point(table.point(p)));
        }
    }
    for (PartitionId id : set.partial) {
        PartitionView pv = path.partition(id);
        EXPECT_TRUE(pv.bounds.intersects(q));
        EXPECT_FALSE(q.contains_rect(pv.bounds));
    }
}

TEST(StaticKd, WholeDomainIsAllContained) {
    IndexTable table = make_grid();
    StaticKdAccessPath path(config(/*leaf_min_size=*/16));
    path.prepare(table);
    path.ensure_built();

    QueryPartitionSet full = path.locate(HyperRect{{{-1.0, 17.0}, {-1.0, 17.0}}});
    EXPECT_EQ(full.fully_contained.size(), path.active_partitions().size());
    EXPECT_TRUE(full.partial.empty());

    QueryPartitionSet none = path.locate(HyperRect{{{50.0, 60.0}, {50.0, 60.0}}});
    EXPECT_TRUE(none.fully_contained.empty());
    EXPECT_TRUE(none.partial.empty());
}

TEST(StaticKd, RefineIsNoOpAndReturnsFrontier) {
    IndexTable table = make_grid();
    StaticKdAccessPath path(config(/*leaf_min_size=*/16));
    path.prepare(table);
    path.ensure_built();

    auto before = path.active_partitions();
    std::sort(before.begin(), before.end());

    HyperRect q{{{3.0, 11.0}, {3.0, 11.0}}};
    auto rr = path.refine(q, table);

    auto after = path.active_partitions();
    std::sort(after.begin(), after.end());

    EXPECT_TRUE(rr.retired.empty());
    EXPECT_EQ(before, after) << "static refine must not change the structure";

    // The returned frontier equals a fresh locate (no adaptation happened).
    QueryPartitionSet located = path.locate(q);
    auto sorted = [](std::vector<PartitionId> v) {
        std::sort(v.begin(), v.end());
        return v;
    };
    EXPECT_EQ(sorted(rr.frontier.fully_contained), sorted(located.fully_contained));
    EXPECT_EQ(sorted(rr.frontier.partial), sorted(located.partial));
}

TEST(StaticKd, AncestryReachesParentlessRoot) {
    IndexTable table = make_grid();
    StaticKdAccessPath path(config(/*leaf_min_size=*/16));
    path.prepare(table);
    path.ensure_built();

    EXPECT_FALSE(path.parent(0).has_value());
    for (PartitionId id : path.active_partitions()) {
        PartitionId cur = id;
        std::size_t hops = 0;
        while (auto p = path.parent(cur)) {
            cur = *p;
            ASSERT_LT(++hops, table.size());
        }
        EXPECT_EQ(cur, 0u) << "ancestry does not reach the root";
    }
}

TEST(StaticKd, RefineWrongTableThrows) {
    IndexTable table = make_grid();
    IndexTable other = make_grid();
    StaticKdAccessPath path(config(/*leaf_min_size=*/16));
    path.prepare(table);
    path.ensure_built();
    EXPECT_THROW(path.refine(HyperRect{{{3.0, 11.0}, {3.0, 11.0}}}, other),
                 std::invalid_argument);
}

TEST(StaticKd, RangesNotRowIdOrdered) {
    StaticKdAccessPath path(config(/*leaf_min_size=*/16));
    EXPECT_FALSE(path.ranges_are_row_id_ordered());
}

}  // namespace

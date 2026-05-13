// Cracking, partition invariants, and ancestry for the adaptive KD substrate.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "a3i/core/geometry.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/adaptive_kd_access_path.hpp"

namespace {

using namespace a3i;

// A 12 x 12 integer grid of points over the [0, 12) x [0, 12) domain:
// 144 points, row id == (x * 12 + y) in load order.
IndexTable make_grid() {
    std::vector<double> xs;
    std::vector<double> ys;
    for (int x = 0; x < 12; ++x) {
        for (int y = 0; y < 12; ++y) {
            xs.push_back(static_cast<double>(x));
            ys.push_back(static_cast<double>(y));
        }
    }
    return IndexTable::from_columns({xs, ys});
}

HyperRect domain() { return HyperRect{{{0.0, 12.0}, {0.0, 12.0}}}; }

SubstrateConfig config(std::uint32_t threshold) {
    SubstrateConfig cfg;
    cfg.domain_bounds = domain();
    cfg.refinement_threshold = threshold;
    return cfg;
}

// Active partitions are disjoint and cover [0, N); ids are unique.
void check_cover(const AdaptiveAccessPath& path, std::size_t n) {
    auto active = path.active_partitions();
    std::vector<bool> covered(n, false);
    std::vector<PartitionId> ids;
    for (PartitionId id : active) {
        PartitionView pv = path.partition(id);
        EXPECT_TRUE(pv.active);
        EXPECT_LE(pv.begin, pv.end);
        ids.push_back(id);
        for (IndexPos p = pv.begin; p < pv.end; ++p) {
            ASSERT_FALSE(covered[p]) << "position " << p << " double-covered";
            covered[p] = true;
        }
    }
    for (std::size_t p = 0; p < n; ++p) EXPECT_TRUE(covered[p]) << "position " << p;
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end()) << "duplicate id";
}

// Every point inside an active leaf lies within that leaf's bounds, and the
// row ids across all positions are exactly {0 .. N-1} (no loss or duplication
// from the swaps carrying the row_id payload).
void check_points_in_bounds(const AdaptiveAccessPath& path, const IndexTable& table) {
    for (PartitionId id : path.active_partitions()) {
        PartitionView pv = path.partition(id);
        for (IndexPos p = pv.begin; p < pv.end; ++p) {
            EXPECT_TRUE(pv.bounds.contains_point(table.point(p)))
                << "position " << p << " outside partition " << id << " bounds";
        }
    }
    std::vector<RowId> rows(table.row_ids().begin(), table.row_ids().end());
    std::sort(rows.begin(), rows.end());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        EXPECT_EQ(rows[i], static_cast<RowId>(i)) << "row id payload corrupted";
    }
}

TEST(Partitioner, RefineCracksBoundaryPartition) {
    IndexTable table = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/8));
    path.prepare(table);
    path.ensure_built();
    ASSERT_EQ(path.active_partitions().size(), 1u);

    HyperRect q{{{3.0, 7.0}, {3.0, 7.0}}};
    auto rr = path.refine(q, table);

    EXPECT_FALSE(rr.retired.empty());
    EXPECT_GT(path.active_partitions().size(), 1u);
    check_cover(path, table.size());
    check_points_in_bounds(path, table);

    // The returned frontier must equal what a fresh locate now reports, so
    // the caller never has to descend the structure a second time.
    QueryPartitionSet relocated = path.locate(q);
    auto sorted = [](std::vector<PartitionId> v) {
        std::sort(v.begin(), v.end());
        return v;
    };
    EXPECT_EQ(sorted(rr.frontier.fully_contained),
              sorted(relocated.fully_contained));
    EXPECT_EQ(sorted(rr.frontier.partial), sorted(relocated.partial));
}

TEST(Partitioner, CrackedFullyContainedMatchesQuery) {
    IndexTable table = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/4));
    path.prepare(table);
    path.ensure_built();

    HyperRect q{{{3.0, 7.0}, {3.0, 7.0}}};
    path.refine(q, table);

    // After isolating q, locate must report at least one fully-contained
    // partition, and every fully-contained partition's points satisfy q.
    QueryPartitionSet set = path.locate(q);
    EXPECT_FALSE(set.fully_contained.empty());
    for (PartitionId id : set.fully_contained) {
        PartitionView pv = path.partition(id);
        EXPECT_TRUE(q.contains_rect(pv.bounds));
        for (IndexPos p = pv.begin; p < pv.end; ++p) {
            EXPECT_TRUE(q.contains_point(table.point(p)));
        }
    }
}

TEST(Partitioner, RetiredParentsAreInactiveWithAncestry) {
    IndexTable table = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/4));
    path.prepare(table);
    path.ensure_built();

    HyperRect q{{{3.0, 7.0}, {3.0, 7.0}}};
    auto rr = path.refine(q, table);
    ASSERT_FALSE(rr.retired.empty());

    for (PartitionId id : rr.retired) {
        EXPECT_FALSE(path.partition(id).active) << "retired parent still active";
    }
    // Every active leaf has a parent chain that terminates at the parentless
    // root, and the root itself is now retired (it was the first crack).
    for (PartitionId id : path.active_partitions()) {
        PartitionId cur = id;
        std::size_t hops = 0;
        while (auto p = path.parent(cur)) {
            cur = *p;
            ASSERT_LT(++hops, table.size());
        }
        EXPECT_EQ(cur, 0u) << "ancestry does not reach the root";
    }
    EXPECT_FALSE(path.parent(0).has_value());
}

TEST(Partitioner, FailedCrackLeavesPartitionUnchanged) {
    IndexTable table = make_grid();
    // Domain far larger than the data: all 144 points sit in [0,12) on each
    // axis, but the root's bounds span [0,100).
    SubstrateConfig cfg;
    cfg.domain_bounds = HyperRect{{{0.0, 100.0}, {0.0, 100.0}}};
    cfg.refinement_threshold = 4;
    AdaptiveKdAccessPath path(cfg);
    path.prepare(table);
    path.ensure_built();

    // This query is inside the (large) domain, so the oversized root is a
    // boundary partition and refine attempts to crack it; but every pivot
    // lies above all of the data, so each Hoare partition lands every point
    // on one side and fails. Nothing splits.
    HyperRect q{{{50.0, 60.0}, {50.0, 60.0}}};
    ASSERT_FALSE(path.locate(q).partial.empty());
    auto rr = path.refine(q, table);
    EXPECT_TRUE(rr.retired.empty());
    EXPECT_EQ(path.active_partitions().size(), 1u);
}

TEST(Partitioner, BelowThresholdIsNotCracked) {
    IndexTable table = make_grid();
    // Threshold above the whole table: the single root is never large enough.
    AdaptiveKdAccessPath path(config(/*threshold=*/1000));
    path.prepare(table);
    path.ensure_built();

    HyperRect q{{{3.0, 7.0}, {3.0, 7.0}}};
    auto rr = path.refine(q, table);
    EXPECT_TRUE(rr.retired.empty());
    EXPECT_EQ(path.active_partitions().size(), 1u);
}

TEST(Partitioner, RepeatedRefineConverges) {
    IndexTable table = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/4));
    path.prepare(table);
    path.ensure_built();

    HyperRect q{{{3.0, 7.0}, {3.0, 7.0}}};
    path.refine(q, table);
    std::size_t after_first = path.active_partitions().size();
    path.refine(q, table);
    std::size_t after_second = path.active_partitions().size();

    // The boundary child isolating q is at or below threshold after the first
    // pass, so a second identical refine adds nothing.
    EXPECT_EQ(after_second, after_first);
    check_cover(path, table.size());
    check_points_in_bounds(path, table);
}

TEST(Partitioner, RefineWrongTableThrows) {
    IndexTable table = make_grid();
    IndexTable other = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/4));
    path.prepare(table);
    path.ensure_built();
    EXPECT_THROW(path.refine(HyperRect{{{3.0, 7.0}, {3.0, 7.0}}}, other),
                 std::invalid_argument);
}

}  // namespace

// Tests for the grid substrate and the KdTree flat-partition install it builds
// on: tile coverage, arithmetic tile routing, the row->tile map, and cracking.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "a3i/access_path/partition_view.hpp"
#include "a3i/access_path/substrate_factory.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/grid_akd_access_path.hpp"
#include "a3i/substrates/kd_tree.hpp"

namespace {

using namespace a3i;

// 4x4 lattice: row_id r sits in tile cell (r/4, r%4), so its flat cell == r.
IndexTable lattice_table() {
    std::vector<double> xs, ys;
    for (int r = 0; r < 16; ++r) {
        xs.push_back((r / 4) + 0.5);
        ys.push_back((r % 4) + 0.5);
    }
    return IndexTable::from_columns({xs, ys});
}

SubstrateConfig grid_config(std::uint32_t k, std::uint32_t psize) {
    SubstrateConfig cfg;
    cfg.partition_size = psize;
    cfg.partitions_per_dimension = k;
    cfg.data_bounds = HyperRect{{Range{0.0, 4.0}, Range{0.0, 4.0}}};
    return cfg;
}

TEST(KdTreeForest, ResetWithChildrenInstallsRootAndLeaves) {
    KdTree tree;
    const HyperRect root{{Range{0, 4}, Range{0, 4}}};
    const std::vector<HyperRect> children = {
        HyperRect{{Range{0, 2}, Range{0, 4}}},
        HyperRect{{Range{2, 4}, Range{0, 4}}},
    };
    tree.reset_with_children(root, /*n=*/5, children, /*offsets=*/{0, 3, 5});

    EXPECT_EQ(tree.roots(), (std::vector<PartitionId>{0}));
    EXPECT_FALSE(tree.is_leaf(0));
    EXPECT_TRUE(tree.is_leaf(1));
    EXPECT_TRUE(tree.is_leaf(2));
    EXPECT_FALSE(tree.parent(0).has_value());
    EXPECT_EQ(tree.parent(1), std::optional<PartitionId>(0));
    EXPECT_EQ(tree.parent(2), std::optional<PartitionId>(0));

    const PartitionView p1 = tree.partition(1);
    EXPECT_EQ(p1.begin, 0u);
    EXPECT_EQ(p1.end, 3u);
    EXPECT_TRUE(p1.active);
    const PartitionView p2 = tree.partition(2);
    EXPECT_EQ(p2.begin, 3u);
    EXPECT_EQ(p2.end, 5u);

    // Only the leaves are active; the root is an inactive parent.
    EXPECT_EQ(tree.active_partitions().size(), 2u);
}

TEST(KdTreeForest, ResetWithChildrenRejectsBadOffsets) {
    KdTree tree;
    const HyperRect root{{Range{0, 4}}};
    const std::vector<HyperRect> children = {HyperRect{{Range{0, 4}}}};
    EXPECT_THROW(tree.reset_with_children(root, 5, children, {0}),
                 std::invalid_argument);  // wrong size
    EXPECT_THROW(tree.reset_with_children(root, 5, children, {0, 4}),
                 std::invalid_argument);  // does not end at n
}

TEST(GridAkd, TilesCoverThePartitionDisjointly) {
    IndexTable table = lattice_table();
    GridAkdAccessPath grid(grid_config(/*k=*/4, /*psize=*/16));
    grid.prepare(table);
    grid.ensure_built();

    const auto active = grid.active_partitions();
    EXPECT_EQ(active.size(), 16u);  // one tile per lattice cell

    std::vector<char> seen(16, 0);
    std::uint64_t total = 0;
    for (PartitionId id : active) {
        const PartitionView pv = grid.partition(id);
        for (IndexPos p = pv.begin; p < pv.end; ++p) {
            ASSERT_LT(p, 16u);
            EXPECT_EQ(seen[p], 0);  // disjoint
            seen[p] = 1;
        }
        total += pv.end - pv.begin;
    }
    EXPECT_EQ(total, 16u);  // covering
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](char c) { return c == 1; }));
}

TEST(GridAkd, RowOwnerMapMatchesTiles) {
    IndexTable table = lattice_table();
    GridAkdAccessPath grid(grid_config(4, 16));
    grid.prepare(table);
    grid.ensure_built();

    const std::vector<PartitionId>* owner = grid.row_owner_map();
    ASSERT_NE(owner, nullptr);
    ASSERT_EQ(owner->size(), 16u);
    for (RowId r = 0; r < 16; ++r) {
        EXPECT_EQ((*owner)[r], static_cast<PartitionId>(1 + r));
    }
}

TEST(GridAkd, OverlappingChildrenReturnsExactlyTheTouchedTiles) {
    IndexTable table = lattice_table();
    GridAkdAccessPath grid(grid_config(4, 16));
    grid.prepare(table);
    grid.ensure_built();

    // [0.4, 1.6]^2 falls inside x-cells {0,1} and y-cells {0,1}.
    const HyperRect q{{Range{0.4, 1.6}, Range{0.4, 1.6}}};
    auto got = grid.overlapping_children(0, q);
    std::sort(got.begin(), got.end());
    EXPECT_EQ(got, (std::vector<PartitionId>{1, 2, 5, 6}));

    // The routed set is exactly the tiles that are not disjoint from q.
    for (PartitionId id : grid.active_partitions()) {
        const bool routed = std::find(got.begin(), got.end(), id) != got.end();
        const bool overlaps = grid.classify(id, q) != Containment::Disjoint;
        EXPECT_EQ(routed, overlaps) << "tile " << id;
    }
}

TEST(GridAkd, RootIsAnInactiveParentOverAllTiles) {
    IndexTable table = lattice_table();
    GridAkdAccessPath grid(grid_config(4, 16));
    grid.prepare(table);
    grid.ensure_built();

    EXPECT_EQ(grid.roots(), (std::vector<PartitionId>{0}));
    EXPECT_FALSE(grid.is_leaf(0));
    EXPECT_EQ(grid.children(0).size(), 16u);
    const HyperRect enclosing{{Range{-1.0, 5.0}, Range{-1.0, 5.0}}};
    EXPECT_EQ(grid.classify(0, enclosing), Containment::Contained);
}

TEST(GridAkd, RefineCracksAPopulousTileButNotTheRoot) {
    // A single tile over a spread of points; a small threshold forces cracking.
    std::vector<double> xs, ys;
    for (int i = 0; i < 64; ++i) {
        xs.push_back(i % 8);
        ys.push_back(i / 8);
    }
    IndexTable table = IndexTable::from_columns({xs, ys});
    SubstrateConfig cfg;
    cfg.partition_size = 4;
    cfg.partitions_per_dimension = 1;  // one tile spanning the whole domain
    GridAkdAccessPath grid(cfg);
    grid.prepare(table);
    grid.ensure_built();
    ASSERT_EQ(grid.active_partitions().size(), 1u);

    const HyperRect q{{Range{2.0, 5.0}, Range{2.0, 5.0}}};
    const auto retired = grid.refine(/*tile=*/1, q, table);
    EXPECT_FALSE(retired.empty());
    EXPECT_GT(grid.active_partitions().size(), 1u);

    // The root is never cracked.
    EXPECT_TRUE(grid.refine(0, q, table).empty());
}

}  // namespace

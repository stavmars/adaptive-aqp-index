#include <gtest/gtest.h>

#include <vector>

#include "a3i/aqp/position_bitset.hpp"
#include "a3i/core/types.hpp"

using namespace a3i;

TEST(PositionBitset, StartsEmpty) {
    PositionBitset b(100);
    EXPECT_EQ(b.size(), 100u);
    EXPECT_EQ(b.count(), 0u);
    for (IndexPos p = 0; p < 100; ++p) EXPECT_FALSE(b.contains(p));
}

TEST(PositionBitset, SetReportsNewMembershipAndCounts) {
    PositionBitset b(200);
    EXPECT_TRUE(b.set(0));
    EXPECT_TRUE(b.set(63));
    EXPECT_TRUE(b.set(64));   // crosses a word boundary
    EXPECT_TRUE(b.set(199));
    EXPECT_FALSE(b.set(63));  // already a member
    EXPECT_EQ(b.count(), 4u);
    EXPECT_TRUE(b.contains(0));
    EXPECT_TRUE(b.contains(63));
    EXPECT_TRUE(b.contains(64));
    EXPECT_TRUE(b.contains(199));
    EXPECT_FALSE(b.contains(1));
}

TEST(PositionBitset, EnumeratesMembersAscending) {
    PositionBitset b(300);
    const std::vector<IndexPos> members{5, 64, 65, 130, 299};
    for (IndexPos p : members) b.set(p);

    std::vector<IndexPos> seen;
    b.for_each_set([&](IndexPos p) { seen.push_back(p); });
    EXPECT_EQ(seen, members);
    EXPECT_EQ(b.to_positions(), members);
}

TEST(PositionBitset, NonMultipleOfWordSize) {
    PositionBitset b(10);  // fits in one word, only 10 valid positions
    b.set(0);
    b.set(9);
    EXPECT_EQ(b.count(), 2u);
    EXPECT_EQ(b.to_positions(), (std::vector<IndexPos>{0, 9}));
}

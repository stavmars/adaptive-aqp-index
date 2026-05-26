#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include "a3i/aqp/position_bitset.hpp"
#include "a3i/aqp/stratum_cursor.hpp"
#include "a3i/aqp/summary.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/util/rng.hpp"

using namespace a3i;

namespace {

// Two partitions laid out contiguously in one table. Row ids within each
// partition are intentionally NOT ascending in position order, so a correct
// cursor must sort at construction. Partition A row ids are multiples of 10;
// partition B row ids end in 5 -- lets a merge test check tag routing.
IndexTable make_table() {
    std::vector<RowId> row_ids{50, 10, 30, 20, 40,   // partition A, [0,5)
                               5, 15, 25, 35, 45};    // partition B, [5,10)
    std::vector<double> points(row_ids.size(), 0.0);  // 1-D, values unused here
    return IndexTable(std::move(points), std::move(row_ids), /*dimensions=*/1);
}

// Drain a merge fully in the given chunk size, returning (row_id, tag) pairs.
std::vector<std::pair<RowId, StratumTag>> drain(KWayMerge& merge,
                                                std::size_t chunk) {
    std::vector<std::pair<RowId, StratumTag>> out;
    std::vector<RowId> ids(chunk);
    std::vector<StratumTag> tags(chunk);
    while (!merge.empty()) {
        const std::size_t n = merge.next_chunk(ids, tags);
        for (std::size_t i = 0; i < n; ++i) out.emplace_back(ids[i], tags[i]);
    }
    return out;
}

}  // namespace

TEST(StratumCursor, ReusableFullSortsAndMarksTracker) {
    const IndexTable table = make_table();
    SampleTracker tracker(5);
    StratumCursor cur =
        make_reusable_full_cursor(table, /*begin=*/0, /*size=*/5, tracker, 7);

    EXPECT_EQ(cur.tag, 7);
    EXPECT_EQ(cur.owned, (std::vector<RowId>{10, 20, 30, 40, 50}));
    EXPECT_EQ(tracker.count(), 5u);
    for (IndexPos p = 0; p < 5; ++p) EXPECT_TRUE(tracker.contains(p));
}

TEST(StratumCursor, SampledMarksTrackerAndSorts) {
    const IndexTable table = make_table();
    SampleTracker tracker(5);
    Rng rng(123);
    StratumCursor cur = make_reusable_sampled_cursor(table, /*begin=*/0,
                                                     /*size=*/5, tracker,
                                                     /*count=*/3, rng, 0);

    EXPECT_EQ(cur.owned.size(), 3u);
    EXPECT_TRUE(std::is_sorted(cur.owned.begin(), cur.owned.end()));
    EXPECT_EQ(tracker.count(), 3u);
}

// Cumulative: a second sampled round draws only untracked positions.
TEST(StratumCursor, SampledRoundsAreCumulative) {
    const IndexTable table = make_table();
    SampleTracker tracker(5);
    Rng rng(55);
    StratumCursor r1 =
        make_reusable_sampled_cursor(table, 0, 5, tracker, 2, rng, 0);
    StratumCursor r2 =
        make_reusable_sampled_cursor(table, 0, 5, tracker, 2, rng, 0);

    EXPECT_EQ(tracker.count(), 4u);
    std::set<RowId> all(r1.owned.begin(), r1.owned.end());
    for (RowId id : r2.owned) {
        EXPECT_TRUE(all.insert(id).second) << "round 2 re-read row " << id;
    }
}

TEST(StratumCursor, QueryLocalFullReadsQualifyingPositions) {
    const IndexTable table = make_table();
    SampleTracker tracker(5);
    PositionBitset qualifying(5);  // partition-local positions 0, 2, 4
    qualifying.set(0);             // row id 50
    qualifying.set(2);             // row id 30
    qualifying.set(4);             // row id 40
    StratumCursor cur =
        make_query_local_full_cursor(table, 0, qualifying, tracker, 1);

    EXPECT_EQ(cur.owned, (std::vector<RowId>{30, 40, 50}));
    EXPECT_EQ(tracker.count(), 3u);
    EXPECT_TRUE(tracker.contains(0));
    EXPECT_TRUE(tracker.contains(2));
    EXPECT_TRUE(tracker.contains(4));
}

// The headline merge property: ascending gather order across multiple strata.
TEST(StratumCursor, KWayMergeYieldsAscendingAcrossStrata) {
    const IndexTable table = make_table();
    SampleTracker ta(5), tb(5);
    StratumCursor a = make_reusable_full_cursor(table, /*begin=*/0, 5, ta, 0);
    StratumCursor b = make_reusable_full_cursor(table, /*begin=*/5, 5, tb, 1);

    std::vector<StratumCursor> cursors{std::move(a), std::move(b)};
    KWayMerge merge(cursors);
    const auto seq = drain(merge, /*chunk=*/3);

    ASSERT_EQ(seq.size(), 10u);
    for (std::size_t i = 1; i < seq.size(); ++i) {
        EXPECT_LE(seq[i - 1].first, seq[i].first) << "not ascending at " << i;
    }
    // Tag routing: partition A ids are multiples of 10, B ids end in 5.
    for (const auto& [id, tag] : seq) {
        if (id % 10 == 0) {
            EXPECT_EQ(tag, 0);
        } else {
            EXPECT_EQ(tag, 1);
        }
    }
}

TEST(StratumCursor, MergeIsInvariantToChunkSize) {
    const IndexTable table = make_table();
    SampleTracker a1(5), b1(5), a2(5), b2(5);

    StratumCursor a = make_reusable_full_cursor(table, 0, 5, a1, 0);
    StratumCursor b = make_reusable_full_cursor(table, 5, 5, b1, 1);
    std::vector<StratumCursor> small{std::move(a), std::move(b)};
    KWayMerge merge_small(small);
    const auto seq_small = drain(merge_small, /*chunk=*/1);

    StratumCursor c = make_reusable_full_cursor(table, 0, 5, a2, 0);
    StratumCursor d = make_reusable_full_cursor(table, 5, 5, b2, 1);
    std::vector<StratumCursor> big{std::move(c), std::move(d)};
    KWayMerge merge_big(big);
    const auto seq_big = drain(merge_big, /*chunk=*/100);

    EXPECT_EQ(seq_small, seq_big);
}

TEST(StratumCursor, EmptyCursorIsNotMerged) {
    const IndexTable table = make_table();
    SampleTracker ta(5), tb(5);
    Rng rng(1);
    // A zero-delta round produces an empty (already done) cursor.
    StratumCursor empty =
        make_reusable_sampled_cursor(table, 0, 5, ta, /*count=*/0, rng, 0);
    StratumCursor full = make_reusable_full_cursor(table, 5, 5, tb, 1);

    EXPECT_TRUE(empty.done());
    std::vector<StratumCursor> cursors{std::move(empty), std::move(full)};
    KWayMerge merge(cursors);
    const auto seq = drain(merge, /*chunk=*/4);

    ASSERT_EQ(seq.size(), 5u);  // only partition B
    for (const auto& [id, tag] : seq) EXPECT_EQ(tag, 1);
}

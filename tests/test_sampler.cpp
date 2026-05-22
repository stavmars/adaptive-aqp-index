#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

#include "a3i/aqp/sampler.hpp"
#include "a3i/aqp/summary.hpp"
#include "a3i/util/rng.hpp"

using namespace a3i;

namespace {

// Mark every drawn position in the tracker (what a cursor does after a draw).
void mark(SampleTracker& tracker, const std::vector<IndexPos>& positions) {
    for (IndexPos p : positions) tracker.add(p);
}

}  // namespace

TEST(Sampler, DrawsRequestedCountWithinRange) {
    SampleTracker tracker(100);
    Rng rng(12345);
    const auto draw = Sampler::draw_from_range(100, tracker, 10, rng);
    EXPECT_EQ(draw.size(), 10u);
    for (IndexPos p : draw) EXPECT_LT(p, 100u);
}

TEST(Sampler, NoDuplicatesWithinSingleDraw) {
    SampleTracker tracker(1000);
    Rng rng(7);
    const auto draw = Sampler::draw_from_range(1000, tracker, 250, rng);
    const std::set<IndexPos> distinct(draw.begin(), draw.end());
    EXPECT_EQ(distinct.size(), draw.size());
}

// I6: cumulative rounds never re-draw a row.
TEST(Sampler, CumulativeRoundsDoNotOverlap) {
    SampleTracker tracker(200);
    Rng rng(99);

    const auto round1 = Sampler::draw_from_range(200, tracker, 40, rng);
    mark(tracker, round1);
    const auto round2 = Sampler::draw_from_range(200, tracker, 40, rng);
    mark(tracker, round2);

    std::set<IndexPos> all(round1.begin(), round1.end());
    for (IndexPos p : round2) {
        EXPECT_TRUE(all.insert(p).second) << "round 2 re-drew position " << p;
    }
    EXPECT_EQ(all.size(), 80u);
    EXPECT_EQ(tracker.count(), 80u);
}

// I12: same seed => identical draw; different seed => (almost surely) different.
TEST(Sampler, DeterministicFromSeed) {
    SampleTracker t1(500), t2(500), t3(500);
    Rng a(42), b(42), c(43);
    const auto da = Sampler::draw_from_range(500, t1, 50, a);
    const auto db = Sampler::draw_from_range(500, t2, 50, b);
    const auto dc = Sampler::draw_from_range(500, t3, 50, c);
    EXPECT_EQ(da, db);
    EXPECT_NE(da, dc);
}

TEST(Sampler, ClampsCountToEligible) {
    SampleTracker tracker(10);
    Rng rng(1);
    const auto draw = Sampler::draw_from_range(10, tracker, 25, rng);
    EXPECT_EQ(draw.size(), 10u);
    const std::set<IndexPos> distinct(draw.begin(), draw.end());
    EXPECT_EQ(distinct.size(), 10u);
}

TEST(Sampler, ExhaustsEligibleAcrossRounds) {
    SampleTracker tracker(30);
    Rng rng(3);
    auto round1 = Sampler::draw_from_range(30, tracker, 20, rng);
    mark(tracker, round1);
    auto round2 = Sampler::draw_from_range(30, tracker, 20, rng);  // only 10 left
    mark(tracker, round2);
    EXPECT_EQ(round2.size(), 10u);
    EXPECT_EQ(tracker.count(), 30u);

    const auto round3 = Sampler::draw_from_range(30, tracker, 5, rng);
    EXPECT_TRUE(round3.empty());
}

TEST(Sampler, DrawFromBitsetStaysWithinCandidates) {
    const std::vector<IndexPos> candidates{2, 5, 9, 11, 14, 20, 21};
    SampleTracker tracker(30);
    Rng rng(8);
    const auto draw = Sampler::draw_from_bitset(candidates, tracker, 4, rng);
    EXPECT_EQ(draw.size(), 4u);
    const std::set<IndexPos> cand(candidates.begin(), candidates.end());
    for (IndexPos p : draw) EXPECT_TRUE(cand.count(p)) << p << " not a candidate";
}

TEST(Sampler, DrawFromBitsetExcludesTracked) {
    const std::vector<IndexPos> candidates{2, 5, 9, 11, 14, 20, 21};
    SampleTracker tracker(30);
    tracker.add(5);
    tracker.add(14);
    Rng rng(8);
    const auto draw =
        Sampler::draw_from_bitset(candidates, tracker, 10, rng);  // clamps
    EXPECT_EQ(draw.size(), 5u);  // 7 candidates minus 2 tracked
    for (IndexPos p : draw) {
        EXPECT_NE(p, 5u);
        EXPECT_NE(p, 14u);
    }
}

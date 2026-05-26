#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

#include "a3i/aqp/position_bitset.hpp"
#include "a3i/aqp/sampler.hpp"
#include "a3i/aqp/summary.hpp"
#include "a3i/util/rng.hpp"

using namespace a3i;

namespace {

void mark(SampleTracker& tracker, const std::vector<IndexPos>& positions) {
    for (IndexPos p : positions) tracker.add(p);
}

// A qualifying bitset over [0, size) marking every position in `members`.
PositionBitset make_qualifying(std::uint64_t size,
                               const std::vector<IndexPos>& members) {
    PositionBitset b(size);
    for (IndexPos p : members) b.set(p);
    return b;
}

}  // namespace

// --- full-range universe (qualifying == nullptr) -------------------------

TEST(Sampler, DrawsRequestedCountWithinRange) {
    SampleTracker tracker(100);
    Rng rng(12345);
    const auto draw = Sampler::draw({100, nullptr}, tracker, 10, rng);
    EXPECT_EQ(draw.size(), 10u);
    for (IndexPos p : draw) EXPECT_LT(p, 100u);
}

TEST(Sampler, NoDuplicatesWithinSingleDraw) {
    SampleTracker tracker(1000);
    Rng rng(7);
    // A large draw relative to the universe takes the enumerate path.
    const auto draw = Sampler::draw({1000, nullptr}, tracker, 600, rng);
    const std::set<IndexPos> distinct(draw.begin(), draw.end());
    EXPECT_EQ(distinct.size(), draw.size());
}

// I6: cumulative rounds never re-draw a row.
TEST(Sampler, CumulativeRoundsDoNotOverlap) {
    SampleTracker tracker(200);
    Rng rng(99);

    const auto round1 = Sampler::draw({200, nullptr}, tracker, 40, rng);
    mark(tracker, round1);
    const auto round2 = Sampler::draw({200, nullptr}, tracker, 40, rng);
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
    const auto da = Sampler::draw({500, nullptr}, t1, 50, a);
    const auto db = Sampler::draw({500, nullptr}, t2, 50, b);
    const auto dc = Sampler::draw({500, nullptr}, t3, 50, c);
    EXPECT_EQ(da, db);
    EXPECT_NE(da, dc);
}

TEST(Sampler, ClampsCountToEligible) {
    SampleTracker tracker(10);
    Rng rng(1);
    const auto draw = Sampler::draw({10, nullptr}, tracker, 25, rng);
    EXPECT_EQ(draw.size(), 10u);
    const std::set<IndexPos> distinct(draw.begin(), draw.end());
    EXPECT_EQ(distinct.size(), 10u);
}

TEST(Sampler, ExhaustsEligibleAcrossRounds) {
    SampleTracker tracker(30);
    Rng rng(3);
    auto round1 = Sampler::draw({30, nullptr}, tracker, 20, rng);
    mark(tracker, round1);
    auto round2 = Sampler::draw({30, nullptr}, tracker, 20, rng);  // only 10 left
    mark(tracker, round2);
    EXPECT_EQ(round2.size(), 10u);
    EXPECT_EQ(tracker.count(), 30u);

    const auto round3 = Sampler::draw({30, nullptr}, tracker, 5, rng);
    EXPECT_TRUE(round3.empty());
}

// --- narrowed universe (qualifying bitset) -------------------------------

TEST(Sampler, DrawStaysWithinQualifying) {
    const std::vector<IndexPos> members{2, 5, 9, 11, 14, 20, 21};
    const PositionBitset qualifying = make_qualifying(30, members);
    SampleTracker tracker(30);
    Rng rng(8);
    const auto draw = Sampler::draw({30, &qualifying}, tracker, 4, rng);
    EXPECT_EQ(draw.size(), 4u);
    const std::set<IndexPos> cand(members.begin(), members.end());
    for (IndexPos p : draw) EXPECT_TRUE(cand.count(p)) << p << " not qualifying";
}

TEST(Sampler, DrawExcludesTracked) {
    const std::vector<IndexPos> members{2, 5, 9, 11, 14, 20, 21};
    const PositionBitset qualifying = make_qualifying(30, members);
    SampleTracker tracker(30);
    tracker.add(5);
    tracker.add(14);
    Rng rng(8);
    const auto draw = Sampler::draw({30, &qualifying}, tracker, 10, rng);  // clamps
    EXPECT_EQ(draw.size(), 5u);  // 7 qualifying minus 2 tracked
    for (IndexPos p : draw) {
        EXPECT_NE(p, 5u);
        EXPECT_NE(p, 14u);
    }
}

// --- both strategies agree on SRSWOR semantics ---------------------------

// A large draw from a small universe takes the enumerate path; a small draw
// from a large universe takes the rejection path. Both must produce a valid
// without-replacement sample: in range, untracked, distinct.
TEST(Sampler, EnumerateAndRejectionBothValidSrswor) {
    // Enumerate regime: draw is a large fraction of the eligible set.
    {
        SampleTracker tracker(50);
        Rng rng(101);
        const auto draw = Sampler::draw({50, nullptr}, tracker, 45, rng);
        EXPECT_EQ(draw.size(), 45u);
        const std::set<IndexPos> distinct(draw.begin(), draw.end());
        EXPECT_EQ(distinct.size(), 45u);
        for (IndexPos p : draw) EXPECT_LT(p, 50u);
    }
    // Rejection regime: tiny draw from a large eligible set.
    {
        SampleTracker tracker(100000);
        Rng rng(101);
        const auto draw = Sampler::draw({100000, nullptr}, tracker, 8, rng);
        EXPECT_EQ(draw.size(), 8u);
        const std::set<IndexPos> distinct(draw.begin(), draw.end());
        EXPECT_EQ(distinct.size(), 8u);
        for (IndexPos p : draw) EXPECT_LT(p, 100000u);
    }
}

// The fully-contained rejection fast path must not enumerate the universe: a
// tiny draw from a very large range completes without materializing it.
TEST(Sampler, FullyContainedTinyDrawUsesRejection) {
    const std::uint64_t huge = 5'000'000;
    SampleTracker tracker(huge);
    Rng rng(2024);
    const auto draw = Sampler::draw({huge, nullptr}, tracker, 5, rng);
    EXPECT_EQ(draw.size(), 5u);
    const std::set<IndexPos> distinct(draw.begin(), draw.end());
    EXPECT_EQ(distinct.size(), 5u);
    for (IndexPos p : draw) EXPECT_LT(p, huge);
}

// Rejection is deterministic per build too.
TEST(Sampler, RejectionPathDeterministicFromSeed) {
    SampleTracker t1(100000), t2(100000);
    Rng a(77), b(77);
    const auto da = Sampler::draw({100000, nullptr}, t1, 8, a);
    const auto db = Sampler::draw({100000, nullptr}, t2, 8, b);
    EXPECT_EQ(da, db);
}

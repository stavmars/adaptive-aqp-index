#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "a3i/aqp/partition_state_store.hpp"
#include "a3i/aqp/prior_stats.hpp"
#include "a3i/aqp/summary.hpp"
#include "a3i/storage/manifest.hpp"

using namespace a3i;

namespace {

constexpr std::size_t kMeasures = 2;

}  // namespace

TEST(PartitionStateStore, RegisterAssignsDenseIds) {
    PartitionStateStore store;
    EXPECT_EQ(store.register_partition(kMeasures), 0u);
    EXPECT_EQ(store.register_partition(kMeasures), 1u);
    EXPECT_EQ(store.register_partition(kMeasures), 2u);
    EXPECT_EQ(store.partition_count(), 3u);
}

TEST(PartitionStateStore, FindReturnsNullWhenAbsent) {
    PartitionStateStore store;
    const PartitionId p = store.register_partition(kMeasures);
    EXPECT_EQ(store.find(p, 0), nullptr);
    EXPECT_EQ(store.find(99, 0), nullptr);  // unknown partition
    EXPECT_EQ(store.find(p, 5), nullptr);   // unknown measure
}

TEST(PartitionStateStore, GetOrCreateSharesOneTrackerAcrossMeasures) {
    PartitionStateStore store;
    const PartitionId p = store.register_partition(kMeasures);

    MeasureSummary& s0 = store.get_or_create(p, 0, /*population=*/50);
    MeasureSummary& s1 = store.get_or_create(p, 1, /*population=*/50);
    EXPECT_EQ(s0.population_size, 50u);
    EXPECT_EQ(s1.population_size, 50u);
    ASSERT_NE(s0.tracker, nullptr);
    EXPECT_EQ(s0.tracker.get(), s1.tracker.get());  // same bitmap
    EXPECT_EQ(s0.tracker->size(), 50u);

    // Re-fetch returns the same object, not a fresh one.
    MeasureSummary& again = store.get_or_create(p, 0, 50);
    EXPECT_EQ(&again, &s0);
}

TEST(PartitionStateStore, UpdateSampledFoldsAndInfersState) {
    PartitionStateStore store;
    const PartitionId p = store.register_partition(kMeasures);
    store.get_or_create(p, 0, /*population=*/10);

    SampleDelta d1;
    d1.new_sampled_rows = 3;
    d1.moments.add_if_present(2.0);
    d1.moments.add_if_present(4.0);
    d1.moments.add_if_present(6.0);
    store.update_sampled(p, 0, d1);

    const MeasureSummary* s = store.find(p, 0);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->sampled_rows, 3u);
    EXPECT_NEAR(s->non_nan.sum(), 12.0, 1e-12);
    EXPECT_TRUE(s->sampled());
    EXPECT_FALSE(s->complete());

    SampleDelta d2;
    d2.new_sampled_rows = 2;
    d2.moments.add_if_present(8.0);
    d2.moments.add_if_present(10.0);
    store.update_sampled(p, 0, d2);
    EXPECT_EQ(s->sampled_rows, 5u);
    EXPECT_NEAR(s->non_nan.sum(), 30.0, 1e-12);
    EXPECT_EQ(s->non_nan.non_nan_count, 5u);
}

TEST(PartitionStateStore, ReplaceWithCompleteMarksExact) {
    PartitionStateStore store;
    const PartitionId p = store.register_partition(kMeasures);

    MeasureSummary complete;
    complete.population_size = 7;
    complete.sampled_rows = 7;
    for (double v : {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}) {
        complete.non_nan.add_if_present(v);
    }
    store.replace_with_complete(p, 0, complete);

    const MeasureSummary* s = store.find(p, 0);
    ASSERT_NE(s, nullptr);
    EXPECT_TRUE(s->complete());
    EXPECT_NEAR(s->non_nan.sum(), 28.0, 1e-12);
}

TEST(PartitionStateStore, RetireKeepsSummaries) {
    PartitionStateStore store;
    const PartitionId p = store.register_partition(kMeasures);
    MeasureSummary& s = store.get_or_create(p, 0, 10);
    s.sampled_rows = 4;

    store.retire_partition(p);
    EXPECT_FALSE(store.is_current(p));
    const MeasureSummary* kept = store.find(p, 0);
    ASSERT_NE(kept, nullptr);
    EXPECT_EQ(kept->sampled_rows, 4u);  // summary retained
}

TEST(PartitionStateStore, EnsurePartitionGrowsStorage) {
    PartitionStateStore store;
    store.ensure_partition(5, kMeasures);
    EXPECT_GE(store.partition_count(), 6u);
    EXPECT_TRUE(store.is_current(5));
    // Slot is usable.
    MeasureSummary& s = store.get_or_create(5, 1, 8);
    EXPECT_EQ(s.population_size, 8u);
}

// ---- PriorStatsProvider --------------------------------------------------

TEST(PriorStatsProvider, PrefersOwnCompleteSummary) {
    PartitionStateStore store;
    const PartitionId p = store.register_partition(kMeasures);
    MeasureSummary complete;
    complete.population_size = 3;
    complete.sampled_rows = 3;
    complete.non_nan.add_if_present(1.0);
    store.replace_with_complete(p, 0, complete);

    std::vector<GlobalMeasureStats> globals(kMeasures);
    PriorStatsProvider prior(
        store, [](PartitionId) { return std::optional<PartitionId>{}; }, globals);

    EXPECT_NE(prior.complete_partition_summary(p, 0), nullptr);
    // A sampled-but-incomplete summary is not "complete".
    store.get_or_create(p, 1, 10).sampled_rows = 2;
    EXPECT_EQ(prior.complete_partition_summary(p, 1), nullptr);
}

TEST(PriorStatsProvider, WalksToNearestRetainedExactAncestor) {
    // Chain: child(2) -> mid(1) -> root(0). Root is exact for measure 0.
    PartitionStateStore store;
    store.register_partition(kMeasures);  // 0 root
    store.register_partition(kMeasures);  // 1 mid
    store.register_partition(kMeasures);  // 2 child

    MeasureSummary exact;
    exact.population_size = 5;
    exact.sampled_rows = 5;
    exact.non_nan.add_if_present(9.0);
    store.replace_with_complete(0, 0, exact);
    store.retire_partition(0);
    store.retire_partition(1);

    auto parent_of = [](PartitionId id) -> std::optional<PartitionId> {
        if (id == 2) return PartitionId{1};
        if (id == 1) return PartitionId{0};
        return std::nullopt;  // root
    };
    std::vector<GlobalMeasureStats> globals(kMeasures);
    PriorStatsProvider prior(store, parent_of, globals);

    const MeasureSummary* anc = prior.nearest_retained_exact_ancestor(2, 0);
    ASSERT_NE(anc, nullptr);
    EXPECT_NEAR(anc->non_nan.sum(), 9.0, 1e-12);

    // No ancestor is exact for measure 1.
    EXPECT_EQ(prior.nearest_retained_exact_ancestor(2, 1), nullptr);
}

TEST(PriorStatsProvider, FallsBackToGlobalStats) {
    PartitionStateStore store;
    store.register_partition(kMeasures);

    std::vector<GlobalMeasureStats> globals(kMeasures);
    globals[0].non_nan_count = 1000;
    globals[0].sum = 5000.0;
    globals[1].sum = -1.0;

    PriorStatsProvider prior(
        store, [](PartitionId) { return std::optional<PartitionId>{}; }, globals);
    EXPECT_EQ(prior.global_measure_stats(0).non_nan_count, 1000u);
    EXPECT_DOUBLE_EQ(prior.global_measure_stats(0).sum, 5000.0);
    EXPECT_THROW(prior.global_measure_stats(9), std::out_of_range);
}

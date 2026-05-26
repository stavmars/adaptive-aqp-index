// Query decomposition: the disjoint contributor split, the exact bucket, and
// the COUNT(*) total, over the adaptive KD substrate on tiny data.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include "a3i/aqp/decompose.hpp"
#include "a3i/aqp/partition_state_store.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/adaptive_kd_access_path.hpp"

namespace {

using namespace a3i;

constexpr std::size_t kMeasures = 2;

// A 12 x 12 integer grid over [0, 12) x [0, 12); row id == x * 12 + y.
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

SubstrateConfig config(std::uint32_t threshold) {
    SubstrateConfig cfg;
    cfg.domain_bounds = HyperRect{{{0.0, 12.0}, {0.0, 12.0}}};
    cfg.refinement_threshold = threshold;
    return cfg;
}

// Ground truth: row ids whose point lies in q (a full scan of the table).
std::set<RowId> qualifying_rows(const IndexTable& table, const HyperRect& q) {
    std::set<RowId> rows;
    for (IndexPos p = 0; p < table.size(); ++p) {
        if (q.contains_point(table.point(p))) rows.insert(table.row_id(p));
    }
    return rows;
}

// Every row id that contributors for measure `mid` cover, inserting into
// `out` and reporting a duplicate (overlap) if one is ever inserted twice.
bool collect_rows_for_measure(const IndexTable& table,
                              const AdaptiveAccessPath& path,
                              const DecompositionResult& d, MeasureId mid,
                              std::set<RowId>& out) {
    bool disjoint = true;
    auto add_range = [&](IndexPos begin, IndexPos end) {
        for (IndexPos p = begin; p < end; ++p) {
            if (!out.insert(table.row_id(p)).second) disjoint = false;
        }
    };
    for (const auto& c : d.decomposition.exact_contributors) {
        if (c.mid != mid) continue;
        const PartitionView pv = path.partition(c.pid);
        add_range(pv.begin, pv.end);
    }
    for (const auto& s : d.decomposition.reusable_strata) {
        if (s.mid != mid) continue;
        add_range(s.begin, s.end);
    }
    for (const auto& s : d.decomposition.query_local_strata) {
        if (s.mid != mid) continue;
        s.qualifying->for_each_set([&](IndexPos local) {
            if (!out.insert(table.row_id(s.begin + local)).second) {
                disjoint = false;
            }
        });
    }
    return disjoint;
}

// Mark every measure of a partition exact, with a known per-measure sum.
void make_partition_exact(PartitionStateStore& store, const AdaptiveAccessPath& path,
                          PartitionId pid, double value_per_row) {
    const PartitionView pv = path.partition(pid);
    const auto population = static_cast<std::uint64_t>(pv.end - pv.begin);
    store.ensure_partition(pid, kMeasures);
    for (MeasureId mid = 0; mid < kMeasures; ++mid) {
        MeasureSummary& s = store.get_or_create(pid, mid, population);
        s.sampled_rows = population;
        for (std::uint64_t i = 0; i < population; ++i) {
            s.non_nan.add_if_present(value_per_row);
        }
    }
}

TEST(Decompose, ContributorsCoverQualifyingRowsDisjointlyPerMeasure) {
    IndexTable table = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/4));
    path.prepare(table);
    path.ensure_built();

    const HyperRect q{{{3.0, 9.0}, {3.0, 9.0}}};
    const RefineResult rr = path.refine(q, table);
    PartitionStateStore store;

    const DecompositionResult d =
        decompose(q, path, rr.frontier, store, table, kMeasures);

    const std::set<RowId> truth = qualifying_rows(table, q);
    EXPECT_EQ(d.total_count, truth.size());

    for (MeasureId mid = 0; mid < kMeasures; ++mid) {
        std::set<RowId> covered;
        const bool disjoint =
            collect_rows_for_measure(table, path, d, mid, covered);
        EXPECT_TRUE(disjoint) << "measure " << mid << " contributors overlap";
        EXPECT_EQ(covered, truth) << "measure " << mid << " coverage mismatch";
    }

    // With no summaries yet, nothing is exact; reusable strata carry the
    // partition's persistent tracker, query-local strata carry none.
    EXPECT_TRUE(d.decomposition.exact_contributors.empty());
    for (const auto& s : d.decomposition.reusable_strata) {
        EXPECT_NE(s.tracker, nullptr);
    }
}

TEST(Decompose, FullyContainedAllExactBecomeExactContributors) {
    IndexTable table = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/4));
    path.prepare(table);
    path.ensure_built();

    const HyperRect q{{{3.0, 9.0}, {3.0, 9.0}}};
    const RefineResult rr = path.refine(q, table);
    ASSERT_FALSE(rr.frontier.fully_contained.empty());

    PartitionStateStore store;
    double expected_sum = 0.0;
    std::uint64_t expected_count = 0;
    for (PartitionId pid : rr.frontier.fully_contained) {
        const PartitionView pv = path.partition(pid);
        const auto pop = static_cast<std::uint64_t>(pv.end - pv.begin);
        make_partition_exact(store, path, pid, /*value_per_row=*/2.0);
        expected_sum += 2.0 * static_cast<double>(pop);
        expected_count += pop;
    }

    const DecompositionResult d =
        decompose(q, path, rr.frontier, store, table, kMeasures);

    // Fully-contained partitions are now exact; none should be a stratum.
    EXPECT_TRUE(d.decomposition.reusable_strata.empty());
    EXPECT_FALSE(d.decomposition.exact_contributors.empty());
    for (MeasureId mid = 0; mid < kMeasures; ++mid) {
        EXPECT_DOUBLE_EQ(d.exact_bucket.sum_by_measure[mid], expected_sum);
        EXPECT_EQ(d.exact_bucket.count_by_measure[mid], expected_count);
    }
    // COUNT(*) is unaffected by which contributor kind covers a row.
    EXPECT_EQ(d.total_count, qualifying_rows(table, q).size());
}

// All-or-nothing: if any requested measure is incomplete, the fully-contained
// partition is sampled (a reusable stratum) for ALL measures, not exact.
TEST(Decompose, AllOrNothingShortCircuit) {
    IndexTable table = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/4));
    path.prepare(table);
    path.ensure_built();

    const HyperRect q{{{3.0, 9.0}, {3.0, 9.0}}};
    const RefineResult rr = path.refine(q, table);
    ASSERT_FALSE(rr.frontier.fully_contained.empty());
    const PartitionId target = rr.frontier.fully_contained.front();

    PartitionStateStore store;
    // Complete only measure 0 of the target; measure 1 stays absent.
    const PartitionView pv = path.partition(target);
    const auto pop = static_cast<std::uint64_t>(pv.end - pv.begin);
    store.ensure_partition(target, kMeasures);
    MeasureSummary& s0 = store.get_or_create(target, 0, pop);
    s0.sampled_rows = pop;
    s0.non_nan.add_if_present(1.0);

    const DecompositionResult d =
        decompose(q, path, rr.frontier, store, table, kMeasures);

    // The target must appear as a reusable stratum (both measures), never as
    // an exact contributor.
    for (const auto& c : d.decomposition.exact_contributors) {
        EXPECT_NE(c.pid, target) << "incomplete partition was treated as exact";
    }
    int reusable_for_target = 0;
    for (const auto& s : d.decomposition.reusable_strata) {
        if (s.pid == target) ++reusable_for_target;
    }
    EXPECT_EQ(reusable_for_target, static_cast<int>(kMeasures));
}

}  // namespace

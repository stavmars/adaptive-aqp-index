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

// Crack `path` toward `q` the same way the descent does: refine every partial
// leaf until the structure stops changing. Used to pre-build the structure
// before marking summaries, so the later decomposition reuses what it finds.
void refine_to_query(AdaptiveAccessPath& path, const HyperRect& q,
                     IndexTable& table) {
    bool progress = true;
    while (progress) {
        progress = false;
        for (PartitionId id : path.active_partitions()) {
            if (path.is_leaf(id) &&
                path.classify(id, q) == Containment::Partial) {
                if (!path.refine(id, q, table).empty()) {
                    progress = true;
                    break;  // the active set changed; restart the scan
                }
            }
        }
    }
}

// Active leaves wholly inside `q`.
std::vector<PartitionId> contained_leaves(const AdaptiveAccessPath& path,
                                          const HyperRect& q) {
    std::vector<PartitionId> out;
    for (PartitionId id : path.active_partitions()) {
        if (path.classify(id, q) == Containment::Contained) out.push_back(id);
    }
    return out;
}

TEST(Decompose, ContributorsCoverQualifyingRowsDisjointlyPerMeasure) {
    IndexTable table = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/4));
    path.prepare(table);
    path.ensure_built();

    const HyperRect q{{{3.0, 9.0}, {3.0, 9.0}}};
    PartitionStateStore store;

    const DecompositionResult d = decompose_descent(
        q, path, store, table, kMeasures, /*persist=*/true, /*allow_refine=*/true);

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
    refine_to_query(path, q, table);
    const std::vector<PartitionId> contained = contained_leaves(path, q);
    ASSERT_FALSE(contained.empty());

    PartitionStateStore store;
    double expected_sum = 0.0;
    std::uint64_t expected_count = 0;
    for (PartitionId pid : contained) {
        const PartitionView pv = path.partition(pid);
        const auto pop = static_cast<std::uint64_t>(pv.end - pv.begin);
        make_partition_exact(store, path, pid, /*value_per_row=*/2.0);
        expected_sum += 2.0 * static_cast<double>(pop);
        expected_count += pop;
    }

    const DecompositionResult d = decompose_descent(
        q, path, store, table, kMeasures, /*persist=*/true, /*allow_refine=*/true);

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
    refine_to_query(path, q, table);
    const std::vector<PartitionId> contained = contained_leaves(path, q);
    ASSERT_FALSE(contained.empty());
    const PartitionId target = contained.front();

    PartitionStateStore store;
    // Complete only measure 0 of the target; measure 1 stays absent.
    const PartitionView pv = path.partition(target);
    const auto pop = static_cast<std::uint64_t>(pv.end - pv.begin);
    store.ensure_partition(target, kMeasures);
    MeasureSummary& s0 = store.get_or_create(target, 0, pop);
    s0.sampled_rows = pop;
    s0.non_nan.add_if_present(1.0);

    const DecompositionResult d = decompose_descent(
        q, path, store, table, kMeasures, /*persist=*/true, /*allow_refine=*/true);

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

// A fully-contained sub-tree whose ancestor already owns a complete summary is
// answered by that one ancestor summary -- contributed once, with its whole
// population -- without descending into (or even visiting) the sub-partitions
// it covers. The top-down descent stops at the highest contained-and-complete
// node directly.
TEST(Decompose, ContainedSubtreeStopsAtHighestCompleteAncestor) {
    IndexTable table = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/4));
    path.prepare(table);
    path.ensure_built();

    // Crack the structure with a boundary query so the root becomes internal
    // and a multi-level sub-tree of active leaves exists.
    const HyperRect crack{{{3.0, 9.0}, {3.0, 9.0}}};
    refine_to_query(path, crack, table);

    // Climb to the root ancestor of an active leaf and mark only it exact.
    PartitionId root = path.active_partitions().front();
    while (path.parent(root)) root = *path.parent(root);
    ASSERT_FALSE(path.is_leaf(root));  // root is internal after cracking

    PartitionStateStore store;
    make_partition_exact(store, path, root, /*value_per_row=*/3.0);

    // Now a whole-domain query: the root is fully contained and complete.
    const HyperRect q{{{0.0, 12.0}, {0.0, 12.0}}};
    const DecompositionResult d = decompose_descent(
        q, path, store, table, kMeasures, /*persist=*/true, /*allow_refine=*/true);

    // The descent stops at the single complete ancestor: one exact contributor
    // per measure, flagged as a retained ancestor, and no strata -- and it does
    // so without visiting any of the leaves under it (only the root node was
    // classified).
    EXPECT_EQ(d.nodes_visited, 1u);
    EXPECT_TRUE(d.decomposition.reusable_strata.empty());
    EXPECT_TRUE(d.decomposition.query_local_strata.empty());
    EXPECT_EQ(d.decomposition.exact_contributors.size(), kMeasures);
    for (const auto& c : d.decomposition.exact_contributors) {
        EXPECT_EQ(c.pid, root);
        EXPECT_TRUE(c.retained_ancestor);
    }
    // COUNT(*) is the whole grid, counted exactly once through the ancestor.
    EXPECT_EQ(d.total_count, static_cast<std::uint64_t>(table.size()));
    const double expected_sum = 3.0 * static_cast<double>(table.size());
    for (MeasureId mid = 0; mid < kMeasures; ++mid) {
        EXPECT_DOUBLE_EQ(d.exact_bucket.sum_by_measure[mid], expected_sum);
        EXPECT_EQ(d.exact_bucket.count_by_measure[mid],
                  static_cast<std::uint64_t>(table.size()));
    }
}

// The mirror of the stop-early case: a contained interior node with NO complete
// summary is descended all the way to its leaves, where the reads (and future
// reusable summaries) land. More than the root node is visited, and every
// contributor is a leaf-level reusable stratum.
TEST(Decompose, ContainedUnsummarizedInteriorDescendsToLeaves) {
    IndexTable table = make_grid();
    AdaptiveKdAccessPath path(config(/*threshold=*/4));
    path.prepare(table);
    path.ensure_built();

    const HyperRect crack{{{3.0, 9.0}, {3.0, 9.0}}};
    refine_to_query(path, crack, table);

    PartitionId root = path.active_partitions().front();
    while (path.parent(root)) root = *path.parent(root);
    ASSERT_FALSE(path.is_leaf(root));

    // No summaries are stored: the whole-domain query cannot stop at the root.
    PartitionStateStore store;
    const HyperRect q{{{0.0, 12.0}, {0.0, 12.0}}};
    const DecompositionResult d = decompose_descent(
        q, path, store, table, kMeasures, /*persist=*/true, /*allow_refine=*/true);

    EXPECT_GT(d.nodes_visited, 1u);  // descended past the root
    EXPECT_TRUE(d.decomposition.exact_contributors.empty());
    EXPECT_FALSE(d.decomposition.reusable_strata.empty());
    // Reads land on the leaves: every reusable stratum is an active leaf.
    for (const auto& s : d.decomposition.reusable_strata) {
        EXPECT_TRUE(path.is_leaf(s.pid));
    }
    EXPECT_EQ(d.total_count, static_cast<std::uint64_t>(table.size()));
}

}  // namespace

#include "a3i/aqp/eager_materialize.hpp"

#include <algorithm>
#include <optional>
#include <span>
#include <vector>

#include "a3i/aqp/outlier_scorer.hpp"
#include "a3i/aqp/summary.hpp"

namespace a3i {

void materialize_all_summaries(const AdaptiveAccessPath& access_path,
                               const IndexTable& table,
                               const BinaryColumnStore& store,
                               PartitionStateStore& state,
                               std::size_t measure_count,
                               const std::vector<PartitionId>* owner_in,
                               OutlierScorer* scorer) {
    const std::vector<PartitionId> active = access_path.active_partitions();
    if (active.empty() || measure_count == 0) return;

    const std::size_t n = table.size();

    // Largest active id, so the per-partition arrays index densely.
    PartitionId max_id = 0;
    for (PartitionId id : active) {
        if (id > max_id) max_id = id;
    }

    // Map each row to the active partition that owns it, plus each partition's
    // population. The caller may supply the row->partition map directly (built
    // during the substrate's own partitioning); otherwise it is derived in one
    // pass over the partition ranges. No measures are read here either way, and
    // the population always comes from the ranges.
    const bool have_owner = owner_in != nullptr && owner_in->size() == n;
    std::vector<PartitionId> owner_local;
    std::vector<std::uint64_t> population(static_cast<std::size_t>(max_id) + 1, 0);
    for (PartitionId id : active) {
        const PartitionView pv = access_path.partition(id);
        population[id] = static_cast<std::uint64_t>(pv.end - pv.begin);
    }
    if (!have_owner) {
        owner_local.assign(n, 0);
        for (PartitionId id : active) {
            const PartitionView pv = access_path.partition(id);
            for (IndexPos pos = pv.begin; pos < pv.end; ++pos) {
                owner_local[table.row_id(pos)] = id;
            }
        }
    }
    const std::vector<PartitionId>& owner = have_owner ? *owner_in : owner_local;

    // Per-partition, per-measure exact moments, in a flat contiguous block
    // indexed [partition * measure_count + measure]. The accumulator access
    // pattern is dictated by the source's row order (owner[r] hops between
    // partitions whenever consecutive rows belong to different leaves), so the
    // update must cost one cache line.
    //
    // A single front-to-back sweep folds ALL measures of a row at once: the
    // row's accumulators are adjacent in the flat block (touched together),
    // and each measure column is consumed as a sequential stream of blocks —
    // a zero-copy view when the column is resident, sequential reads the
    // kernel's readahead turns into a full-bandwidth scan when it is on disk.
    constexpr std::size_t kBlockRows = std::size_t{1} << 20;
    const std::size_t k = measure_count;
    std::vector<MomentStats> acc((static_cast<std::size_t>(max_id) + 1) * k);
    std::vector<std::vector<double>>     scratch(k);
    std::vector<std::span<const double>> cols(k);
    for (std::size_t mid = 0; mid < k; ++mid) {
        store.advise_sequential(static_cast<MeasureId>(mid));
    }
    // Scratch holding one row's measure values, reused per row to feed the
    // scorer without per-row allocation.
    std::vector<double> row_vals(scorer != nullptr ? k : 0);
    for (std::size_t block = 0; block < n; block += kBlockRows) {
        const std::size_t count = std::min(kBlockRows, n - block);
        for (std::size_t mid = 0; mid < k; ++mid) {
            cols[mid] = store.read_rows(static_cast<MeasureId>(mid),
                                        static_cast<RowId>(block), count,
                                        scratch[mid]);
        }
        for (std::size_t i = 0; i < count; ++i) {
            const RowId r = static_cast<RowId>(block + i);
            MomentStats* row_acc = &acc[static_cast<std::size_t>(owner[r]) * k];
            for (std::size_t mid = 0; mid < k; ++mid) {
                row_acc[mid].add_if_present(cols[mid][i]);
            }
            if (scorer != nullptr) {
                for (std::size_t mid = 0; mid < k; ++mid) row_vals[mid] = cols[mid][i];
                scorer->observe(r, row_vals);
            }
        }
    }

    // Store each active partition's summary as complete (every row accounted
    // for), so the decomposition treats a fully-contained active partition as an
    // exact contributor.
    for (PartitionId id : active) {
        const std::uint64_t pop = population[id];
        state.ensure_partition(id, measure_count);
        for (std::size_t mid = 0; mid < measure_count; ++mid) {
            const auto m = static_cast<MeasureId>(mid);
            MeasureSummary& s = state.get_or_create(id, m, pop);
            s.sampled_rows = pop;
            s.non_nan = acc[static_cast<std::size_t>(id) * k + mid];
            // Materialized summaries already fold every row (including flagged
            // ones) into non_nan, so nothing is held out here.
            s.outlier_rows = 0;
            s.outliers_materialized = true;
        }
    }

    // Fold each active partition's exact moments up into its parent chain so
    // every parent partition also carries a complete summary covering its whole
    // sub-tree. Welford merge is associative and commutative, so accumulating
    // each descendant into a parent one at a time yields the parent's exact
    // moments; no measure is read again. A parent is always created before its
    // children, so its id is smaller than any descendant leaf id and the dense
    // accumulators sized to max_id+1 index it directly -- no hashing.
    std::vector<MomentStats> parent_acc(
        (static_cast<std::size_t>(max_id) + 1) * k);
    std::vector<std::uint64_t> parent_pop(static_cast<std::size_t>(max_id) + 1, 0);
    std::vector<char> is_parent(static_cast<std::size_t>(max_id) + 1, 0);
    for (PartitionId id : active) {
        std::optional<PartitionId> cur = access_path.parent(id);
        while (cur) {
            is_parent[*cur] = 1;
            for (std::size_t mid = 0; mid < k; ++mid) {
                parent_acc[static_cast<std::size_t>(*cur) * k + mid].merge(
                    acc[static_cast<std::size_t>(id) * k + mid]);
            }
            parent_pop[*cur] += population[id];
            cur = access_path.parent(*cur);
        }
    }
    for (std::size_t id = 0; id < is_parent.size(); ++id) {
        if (!is_parent[id]) continue;
        const auto pid = static_cast<PartitionId>(id);
        const std::uint64_t pop = parent_pop[id];
        state.ensure_partition(pid, measure_count);
        for (std::size_t mid = 0; mid < k; ++mid) {
            const auto m = static_cast<MeasureId>(mid);
            MeasureSummary& s = state.get_or_create(pid, m, pop);
            s.sampled_rows = pop;
            s.non_nan = parent_acc[id * k + mid];
            s.outlier_rows = 0;
            s.outliers_materialized = true;
        }
    }
}

}  // namespace a3i


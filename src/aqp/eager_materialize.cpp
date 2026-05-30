#include "a3i/aqp/eager_materialize.hpp"

#include <optional>
#include <unordered_map>
#include <vector>

#include "a3i/aqp/summary.hpp"

namespace a3i {

void materialize_all_summaries(const AdaptiveAccessPath& access_path,
                               const IndexTable& table,
                               const BinaryColumnStore& store,
                               PartitionStateStore& state,
                               std::size_t measure_count) {
    const std::vector<PartitionId> active = access_path.active_partitions();
    if (active.empty() || measure_count == 0) return;

    const std::size_t n = table.size();

    // Map each row to the active partition that owns it, in one pass over the
    // partition ranges (no measure reads). Sized to the largest active id so it
    // indexes densely.
    PartitionId max_id = 0;
    for (PartitionId id : active) {
        if (id > max_id) max_id = id;
    }
    std::vector<PartitionId> owner(n, 0);
    std::vector<std::uint64_t> population(static_cast<std::size_t>(max_id) + 1, 0);
    for (PartitionId id : active) {
        const PartitionView pv = access_path.partition(id);
        for (IndexPos pos = pv.begin; pos < pv.end; ++pos) {
            owner[table.row_id(pos)] = id;
        }
        population[id] = static_cast<std::uint64_t>(pv.end - pv.begin);
    }

    // Per-partition, per-measure exact moments. One sequential sweep per column
    // routes each value into its owning partition's accumulator.
    std::vector<std::vector<MomentStats>> acc(
        static_cast<std::size_t>(max_id) + 1,
        std::vector<MomentStats>(measure_count));
    for (std::size_t mid = 0; mid < measure_count; ++mid) {
        const auto m = static_cast<MeasureId>(mid);
        for (RowId r = 0; r < n; ++r) {
            acc[owner[r]][mid].add_if_present(store.measure_value(r, m));
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
            s.non_nan = acc[id][mid];
        }
    }

    // Fold each active partition's exact moments up into its parent chain so
    // every parent partition also carries a complete summary covering its whole
    // sub-tree. Welford merge is associative and commutative, so accumulating
    // each descendant into a parent one at a time yields the parent's exact
    // moments; no measure is read again.
    std::unordered_map<PartitionId, std::vector<MomentStats>> parent_acc;
    std::unordered_map<PartitionId, std::uint64_t> parent_pop;
    for (PartitionId id : active) {
        std::optional<PartitionId> cur = access_path.parent(id);
        while (cur) {
            auto it = parent_acc.find(*cur);
            if (it == parent_acc.end()) {
                it = parent_acc
                         .emplace(*cur, std::vector<MomentStats>(measure_count))
                         .first;
            }
            for (std::size_t mid = 0; mid < measure_count; ++mid) {
                it->second[mid].merge(acc[id][mid]);
            }
            parent_pop[*cur] += population[id];
            cur = access_path.parent(*cur);
        }
    }
    for (auto& [id, moments] : parent_acc) {
        const std::uint64_t pop = parent_pop[id];
        state.ensure_partition(id, measure_count);
        for (std::size_t mid = 0; mid < measure_count; ++mid) {
            const auto m = static_cast<MeasureId>(mid);
            MeasureSummary& s = state.get_or_create(id, m, pop);
            s.sampled_rows = pop;
            s.non_nan = moments[mid];
        }
    }
}

}  // namespace a3i


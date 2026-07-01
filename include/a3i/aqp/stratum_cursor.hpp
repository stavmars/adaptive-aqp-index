// Per-stratum read cursors and the k-way merge that feeds the read path.
//
// A sampling round reads measure values for a set of strata. Each stratum
// contributes one cursor: a sorted-ascending list of row ids to read this
// round, plus a tag that routes each gathered value back to the right
// accumulator. All four kinds of stratum reduce to the same shape -- "produce
// a sorted ascending row-id vector" -- they differ only in how the positions
// are chosen at construction:
//
//   * reusable, full read   -- every position in the partition (exactify);
//   * reusable, sampled read -- `count` new positions drawn without
//                               replacement, recorded in the partition's
//                               persistent tracker;
//   * query-local, full read -- every qualifying position;
//   * query-local, sampled read -- `count` new positions drawn from the
//                               qualifying set, recorded in the query-local
//                               tracker.
//
// In every case the chosen positions are looked up to row ids and sorted at
// construction (the in-memory table is not row-id ordered after in-place
// cracking, so ascending position order is not ascending row-id order). The
// without-replacement bookkeeping is fused into construction: the tracker is
// read to know what is eligible and written to mark what was picked, once,
// here -- peek()/advance() are pure walks over the prepared vector.
//
// The cursors are fed to a min-heap k-way merge keyed on ascending row id, so
// the round's reads sweep each measure column forward in chunks.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "a3i/aqp/position_bitset.hpp"
#include "a3i/aqp/summary.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/util/rng.hpp"

namespace a3i {

struct StratumCursor {
    std::vector<RowId> owned;     // row ids to read; ascending unless gather
                                  // ordering is disabled at construction
    std::size_t        pos = 0;   // walk position into owned
    StratumTag         tag = 0;   // routes gathered values to accumulators
    bool               is_outlier = false;  // held-out rows -> bank, not sample

    RowId peek() const { return owned[pos]; }
    void  advance() { ++pos; }
    bool  done() const { return pos == owned.size(); }
};

/// `sort_owned` controls whether the cursor's row ids are sorted ascending at
/// construction (gather locality); see EngineConfig::sort_gather_by_row_id.

/// Reusable stratum, full read: reads all of [0, size) and marks every
/// position in the persistent tracker.
StratumCursor make_reusable_full_cursor(const IndexTable& table, IndexPos begin,
                                        std::uint64_t size,
                                        SampleTracker& tracker, StratumTag tag,
                                        bool sort_owned = true);

/// Reusable stratum, sampled read: draws `count` new positions without
/// replacement from [0, size) \ tracker \ excluded, marks them in the tracker.
/// `excluded` (partition-local positions, or null) is removed from the universe
/// so those rows are never drawn into the sample.
StratumCursor make_reusable_sampled_cursor(const IndexTable& table,
                                           IndexPos begin, std::uint64_t size,
                                           SampleTracker& tracker,
                                           std::uint64_t count, Rng& rng,
                                           StratumTag tag,
                                           bool sort_owned = true,
                                           const PositionBitset* excluded = nullptr);

/// Query-local stratum, full read: reads every qualifying position and marks
/// each in the query-local tracker. Qualifying positions are partition-local
/// (offsets from `begin`); the tracker shares that coordinate space.
StratumCursor make_query_local_full_cursor(const IndexTable& table,
                                           IndexPos begin,
                                           const PositionBitset& qualifying,
                                           SampleTracker& tracker,
                                           StratumTag tag,
                                           bool sort_owned = true);

/// Query-local stratum, sampled read: draws `count` new positions without
/// replacement from the qualifying set minus the tracker, marks them in the
/// tracker.
StratumCursor make_query_local_sampled_cursor(
    const IndexTable& table, IndexPos begin, const PositionBitset& qualifying,
    SampleTracker& tracker, std::uint64_t count, Rng& rng, StratumTag tag,
    bool sort_owned = true);

/// Held-out (outlier) rows of a reusable partition: the flagged positions in
/// [begin, begin+size), looked up to row ids and sorted like any other cursor,
/// marked `is_outlier` so the merge routes them to the bank rather than the
/// sample. No tracker: these rows are read exactly once, never sampled.
StratumCursor make_outlier_cursor(const IndexTable& table, IndexPos begin,
                                  std::uint64_t size, StratumTag tag,
                                  bool sort_owned = true);

/// Min-heap k-way merge over cursors, interleaving their row ids by current
/// head with the originating stratum tag alongside each id. When every cursor
/// was built sorted the merged stream is globally ascending; with sorting
/// disabled the stream is an arbitrary interleaving, which the gather and
/// moment accumulation handle identically. Cursors are referenced by pointer
/// and must outlive the merge; done cursors are dropped.
class KWayMerge {
public:
    explicit KWayMerge(std::span<StratumCursor> cursors);

    bool empty() const noexcept { return heap_.empty(); }

    /// Pop up to ids_out.size() row ids into ids_out, with the matching tags in
    /// tags_out. The order matches the merge (ascending iff the cursors were
    /// built sorted). Returns the number popped. ids_out and tags_out must have
    /// equal length. If `is_outlier_out` is non-empty (then also equal length),
    /// each popped row's originating-cursor is_outlier flag is written there.
    std::size_t next_chunk(std::span<RowId> ids_out,
                           std::span<StratumTag> tags_out,
                           std::span<char> is_outlier_out = {});

private:
    std::vector<StratumCursor*> heap_;  // min-heap keyed on peek()
};

}  // namespace a3i

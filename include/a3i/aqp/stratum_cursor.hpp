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
    std::vector<RowId> owned;     // sorted ascending; built at construction
    std::size_t        pos = 0;   // walk position into owned
    StratumTag         tag = 0;   // routes gathered values to accumulators

    RowId peek() const { return owned[pos]; }
    void  advance() { ++pos; }
    bool  done() const { return pos == owned.size(); }
};

/// Reusable stratum, full read: reads all of [0, size) and marks every
/// position in the persistent tracker.
StratumCursor make_reusable_full_cursor(const IndexTable& table, IndexPos begin,
                                        std::uint64_t size,
                                        SampleTracker& tracker, StratumTag tag);

/// Reusable stratum, sampled read: draws `count` new positions without
/// replacement from [0, size) \ tracker, marks them in the tracker.
StratumCursor make_reusable_sampled_cursor(const IndexTable& table,
                                           IndexPos begin, std::uint64_t size,
                                           SampleTracker& tracker,
                                           std::uint64_t count, Rng& rng,
                                           StratumTag tag);

/// Query-local stratum, full read: reads every qualifying position and marks
/// each in the query-local tracker. Qualifying positions are partition-local
/// (offsets from `begin`); the tracker shares that coordinate space.
StratumCursor make_query_local_full_cursor(const IndexTable& table,
                                           IndexPos begin,
                                           const PositionBitset& qualifying,
                                           SampleTracker& tracker,
                                           StratumTag tag);

/// Query-local stratum, sampled read: draws `count` new positions without
/// replacement from the qualifying set minus the tracker, marks them in the
/// tracker.
StratumCursor make_query_local_sampled_cursor(
    const IndexTable& table, IndexPos begin, const PositionBitset& qualifying,
    SampleTracker& tracker, std::uint64_t count, Rng& rng, StratumTag tag);

/// Min-heap k-way merge over cursors, yielding row ids in ascending order with
/// the originating stratum tag alongside each id. Cursors are referenced by
/// pointer and must outlive the merge; done cursors are dropped.
class KWayMerge {
public:
    explicit KWayMerge(std::span<StratumCursor> cursors);

    bool empty() const noexcept { return heap_.empty(); }

    /// Pop up to ids_out.size() row ids in ascending order into ids_out, with
    /// the matching tags in tags_out. Returns the number popped. ids_out and
    /// tags_out must have equal length.
    std::size_t next_chunk(std::span<RowId> ids_out,
                           std::span<StratumTag> tags_out);

private:
    std::vector<StratumCursor*> heap_;  // min-heap keyed on peek()
};

}  // namespace a3i

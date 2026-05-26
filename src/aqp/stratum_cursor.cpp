#include "a3i/aqp/stratum_cursor.hpp"

#include <algorithm>
#include <cassert>

#include "a3i/aqp/sampler.hpp"

namespace a3i {

namespace {

/// Look up the row id for each stratum-local position, sort ascending, and
/// wrap in a cursor. `begin` is the partition's start in the index table.
StratumCursor build_cursor(const IndexTable& table, IndexPos begin,
                           const std::vector<IndexPos>& positions,
                           StratumTag tag) {
    StratumCursor cursor;
    cursor.tag = tag;
    cursor.owned.reserve(positions.size());
    for (IndexPos p : positions) {
        cursor.owned.push_back(table.row_id(begin + p));
    }
    std::sort(cursor.owned.begin(), cursor.owned.end());
    return cursor;
}

void mark_all(SampleTracker& tracker, const std::vector<IndexPos>& positions) {
    for (IndexPos p : positions) tracker.add(p);
}

struct CursorGreater {
    bool operator()(const StratumCursor* a, const StratumCursor* b) const {
        return a->peek() > b->peek();  // min-heap on row id via std heap ops
    }
};

}  // namespace

StratumCursor make_reusable_full_cursor(const IndexTable& table, IndexPos begin,
                                        std::uint64_t size,
                                        SampleTracker& tracker,
                                        StratumTag tag) {
    std::vector<IndexPos> positions;
    positions.reserve(static_cast<std::size_t>(size));
    for (std::uint64_t p = 0; p < size; ++p) {
        positions.push_back(static_cast<IndexPos>(p));
    }
    mark_all(tracker, positions);
    return build_cursor(table, begin, positions, tag);
}

StratumCursor make_reusable_sampled_cursor(const IndexTable& table,
                                           IndexPos begin, std::uint64_t size,
                                           SampleTracker& tracker,
                                           std::uint64_t count, Rng& rng,
                                           StratumTag tag) {
    std::vector<IndexPos> positions =
        Sampler::draw({size, /*qualifying=*/nullptr}, tracker, count, rng);
    mark_all(tracker, positions);
    return build_cursor(table, begin, positions, tag);
}

StratumCursor make_query_local_full_cursor(const IndexTable& table,
                                           IndexPos begin,
                                           const PositionBitset& qualifying,
                                           SampleTracker& tracker,
                                           StratumTag tag) {
    std::vector<IndexPos> positions = qualifying.to_positions();
    mark_all(tracker, positions);
    return build_cursor(table, begin, positions, tag);
}

StratumCursor make_query_local_sampled_cursor(
    const IndexTable& table, IndexPos begin, const PositionBitset& qualifying,
    SampleTracker& tracker, std::uint64_t count, Rng& rng, StratumTag tag) {
    std::vector<IndexPos> positions =
        Sampler::draw({qualifying.size(), &qualifying}, tracker, count, rng);
    mark_all(tracker, positions);
    return build_cursor(table, begin, positions, tag);
}

KWayMerge::KWayMerge(std::span<StratumCursor> cursors) {
    heap_.reserve(cursors.size());
    for (StratumCursor& c : cursors) {
        if (!c.done()) heap_.push_back(&c);
    }
    std::make_heap(heap_.begin(), heap_.end(), CursorGreater{});
}

std::size_t KWayMerge::next_chunk(std::span<RowId> ids_out,
                                  std::span<StratumTag> tags_out) {
    assert(ids_out.size() == tags_out.size());
    const std::size_t cap = ids_out.size();
    std::size_t n = 0;
    while (n < cap && !heap_.empty()) {
        std::pop_heap(heap_.begin(), heap_.end(), CursorGreater{});
        StratumCursor* c = heap_.back();
        ids_out[n] = c->peek();
        tags_out[n] = c->tag;
        ++n;
        c->advance();
        if (c->done()) {
            heap_.pop_back();
        } else {
            std::push_heap(heap_.begin(), heap_.end(), CursorGreater{});
        }
    }
    return n;
}

}  // namespace a3i

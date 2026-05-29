// The adaptive access-path abstraction.
//
// An access path organizes the index table's points into disjoint,
// contiguous partitions and answers two questions for the query layer:
// which partitions a rectangle touches (locate), and how the structure
// refines under a query (refine). Concrete substrates (an adaptive
// KD-tree, a static KD-tree, etc.) differ only in how they build
// and split; the query engine depends on this interface alone, never on a
// concrete substrate.
//
// Construction is split in two so a substrate's internal build cost is
// charged to the first query rather than to a separate phase:
//   * prepare()      - load time; take the table, record parameters, do no
//                      partitioning and read no measures.
//   * ensure_built() - start of every query; perform the one-time internal
//                      build on the first call, then return immediately.
//
// Partition invariants every substrate must keep: ids are dense and never
// reused; each partition owns a contiguous [begin, end); active partitions
// are pairwise disjoint and their union is the whole table; every position
// lies in exactly one active partition. Retired partitions remain
// queryable through partition()/parent() for ancestry but are inactive.

#pragma once

#include <optional>
#include <vector>

#include "a3i/access_path/partition_view.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/index_table.hpp"

namespace a3i {

class AdaptiveAccessPath {
public:
    virtual ~AdaptiveAccessPath() = default;

    /// Load time: take (non-owning) ownership of the already-built index
    /// table and record parameters. No partitioning, no measure reads.
    virtual void prepare(IndexTable& table) = 0;

    /// Start of every query: one-time internal construction on the first
    /// call, no-op thereafter. Idempotent.
    virtual void ensure_built() = 0;

    /// Classify the active partitions against `q`: fully contained vs
    /// partially overlapping. Disjoint partitions appear in neither list.
    virtual QueryPartitionSet locate(const HyperRect& q) const = 0;

    /// Refine the structure for `q` by splitting boundary partitions whose
    /// size exceeds the substrate's threshold. Returns the ids of parents
    /// that retired in the process (their summaries are retained by the
    /// caller) together with the post-refinement classification of the
    /// active frontier against `q`, so the caller need not walk the
    /// structure again to re-derive what `locate(q)` would now report.
    /// Never reads measures.
    virtual RefineResult refine(const HyperRect& q, IndexTable& table) = 0;

    /// Snapshot of a partition by id; valid for retired ids too
    /// (`active == false`).
    virtual PartitionView partition(PartitionId id) const = 0;

    /// Ids of the currently active (leaf) partitions.
    virtual std::vector<PartitionId> active_partitions() const = 0;

    /// The partition's refinement parent, or nullopt for a root.
    virtual std::optional<PartitionId> parent(PartitionId id) const = 0;

    /// True iff active partitions' positions are in ascending RowId order,
    /// which lets the read-path cursor skip its sort. False for substrates
    /// whose build/split permutes rows (the KD substrates).
    virtual bool ranges_are_row_id_ordered() const = 0;

    /// True iff `refine()` can grow the structure under queries (a cracking
    /// substrate). False for a fully-built structure whose `refine()` is a
    /// no-op. The engine consults this to decide whether to crack or merely
    /// locate, so the read path stays substrate-agnostic.
    virtual bool supports_refine() const = 0;

    /// True iff the whole structure is materialized up front by `ensure_built()`
    /// with a stable, query-independent partitioning. False when partitions are
    /// created lazily by queries. Only a fully-built structure can have every
    /// node's summary precomputed at initialization.
    virtual bool is_fully_built() const = 0;
};

}  // namespace a3i

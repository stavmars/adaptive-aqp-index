// The adaptive access-path abstraction.
//
// An access path organizes the index table's points into disjoint,
// contiguous partitions and exposes the navigation primitives the query
// layer's top-down descent drives: the entry roots, one-level children,
// per-node classification against a query rectangle, and per-node
// refinement (cracking) of a partial partition. Concrete substrates (an
// adaptive KD-tree, a static KD-tree, etc.) differ only in how they build
// and split; the query engine depends on this interface alone, never on a
// concrete substrate.
//
// The substrate is a pure access path: geometry and partitioning only. It
// knows nothing about summaries, sampling, or aggregates. The descent that
// answers a query lives in the query layer, because only it can see the
// per-partition state and decide where a contained node's stored summary lets
// the descent stop early. classify() and children() are therefore pure
// functions of the structure and the query rectangle.
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

    /// Entry points of the descent. The KD substrates have a single root.
    virtual std::vector<PartitionId> roots() const = 0;

    /// The partition's children one level down, in left-to-right order;
    /// empty for a leaf. The inverse of parent().
    virtual std::vector<PartitionId> children(PartitionId id) const = 0;

    /// True iff `id` has no children (== children(id).empty()).
    virtual bool is_leaf(PartitionId id) const = 0;

    /// Classify one partition's bounds against `q`: disjoint, fully
    /// contained, or partially overlapping. Pure geometry.
    virtual Containment classify(PartitionId id, const HyperRect& q) const = 0;

    /// Refine the partial partition `id` for `q` by cracking it when its
    /// population exceeds the substrate's threshold, isolating the query
    /// rectangle so later queries over the same region reuse fully-contained
    /// children. Returns the ids of parents that retired in the process
    /// (their summaries are retained by the caller). A non-adaptive substrate
    /// is a no-op returning {}. Never reads measures. The caller descends into
    /// the resulting children() afterwards.
    virtual std::vector<PartitionId> refine(PartitionId id, const HyperRect& q,
                                            IndexTable& table) = 0;

    /// Snapshot of a partition by id; valid for retired ids too
    /// (`active == false`).
    virtual PartitionView partition(PartitionId id) const = 0;

    /// Ids of the currently active (leaf) partitions.
    virtual std::vector<PartitionId> active_partitions() const = 0;

    /// The partition's refinement parent, or nullopt for a root.
    virtual std::optional<PartitionId> parent(PartitionId id) const = 0;

    /// True iff `refine()` can grow the structure under queries (a cracking
    /// substrate). False for a structure whose `refine()` is a no-op. The
    /// engine consults this to decide whether to crack or merely locate, so the
    /// read path stays substrate-agnostic.
    virtual bool supports_refine() const = 0;

    /// True iff `ensure_built()` produces the working set of partitions up front
    /// (a real partitioning), as opposed to a single root that queries
    /// subdivide. This is the condition under which every partition's summary
    /// can be precomputed at initialization; it is independent of whether those
    /// partitions later refine (see `supports_refine()`).
    virtual bool has_prebuilt_partitions() const = 0;
};

}  // namespace a3i

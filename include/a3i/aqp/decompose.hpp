// Single top-down descent query decomposition.
//
// Starting from the access path's roots, one recursive descent classifies
// each node against the query and splits the query's qualifying objects into
// the disjoint contributors of a query answer, computing for free the
// deterministic exact totals and the COUNT(*) of all qualifying objects.
//
// At each node: a disjoint node is dropped; a fully-contained node that
// already carries a complete summary for every measure stops the descent and
// contributes one exact summary for its whole contiguous range (no descent
// into the covered sub-tree); an unsummarized contained leaf is a reusable
// stratum; an unsummarized contained interior node is descended; a partial
// leaf is cracked (when the substrate refines) and descended, or scanned into
// a query-local stratum when it stays a leaf; a partial interior node is
// descended. Geometry (which positions qualify) is measure-independent, so a
// partition's qualifying set and sample tracker are shared across its measures.

#pragma once

#include <cstddef>

#include "a3i/access_path/adaptive_access_path.hpp"
#include "a3i/aqp/partition_state_store.hpp"
#include "a3i/aqp/query_decomposition.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/storage/index_table.hpp"

namespace a3i {

/// Decompose query `q` by descending `access_path` from its roots. When
/// `allow_refine` is true the descent cracks partial leaves through
/// `access_path.refine`, retiring the resulting parents in `state_store` (when
/// `persist`). When `persist` is true, `state_store` is consulted so a
/// contained-and-complete node stops the descent as an exact contributor and
/// reusable strata carry the partition's persistent tracker; absent summaries
/// are created. When `persist` is false, `state_store` is not consulted for
/// completeness (no early stop), reusable strata get a fresh per-query tracker,
/// and nothing is kept. No measure values are read in either case.
DecompositionResult decompose_descent(const HyperRect& q,
                                      AdaptiveAccessPath& access_path,
                                      PartitionStateStore& state_store,
                                      IndexTable& table,
                                      std::size_t measure_count,
                                      bool persist,
                                      bool allow_refine);

}  // namespace a3i

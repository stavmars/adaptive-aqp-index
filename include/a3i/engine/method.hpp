// Composing a run from two orthogonal axes: a behavior and a substrate.
//
// A run is the one query engine configured by a small behavior preset and
// executed over a substrate chosen independently. The behavior is captured by
// `EngineConfig` (accuracy mode + whether summaries persist); the substrate is
// any `AdaptiveAccessPath`.

#pragma once

#include <string>

#include "a3i/access_path/adaptive_access_path.hpp"
#include "a3i/engine/query_engine.hpp"

namespace a3i {

/// The named behaviors that compose with any substrate. The oracle full scan
/// is not here: it bypasses the engine and has no substrate.
enum class Behavior { Plain, Agg, Sampling, A3i };

/// The substrate-free `EngineConfig` for a behavior. `allocator` tunes the
/// sampling behaviors and is ignored by the exact ones.
EngineConfig behavior_config(Behavior behavior, AllocatorConfig allocator = {});

/// The run id: the substrate id with the behavior appended (the plain behavior
/// is just the bare substrate id). e.g. ("adkd", Behavior::Agg) -> "adkd_agg".
std::string run_id(const std::string& substrate_id, Behavior behavior);

/// A behavior composed with a substrate, with the two substrate-derived flags
/// filled in. The flags are computed here, never hand-set, so a combination
/// like eager materialization over a lazily-cracked substrate is impossible to
/// construct.
struct ResolvedRunConfig {
    EngineConfig        behavior;
    AdaptiveAccessPath* substrate = nullptr;
    /// = substrate->supports_refine(): may the structure be cracked per query.
    bool allow_refine = false;
    /// = persist_summaries && substrate->is_fully_built(): may every node's
    /// summary be precomputed at initialization (only on a fully-built tree).
    bool eager_materialize = false;

    static ResolvedRunConfig resolve(EngineConfig behavior,
                                     AdaptiveAccessPath& substrate);
};

}  // namespace a3i

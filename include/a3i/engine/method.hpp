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

/// Whether a behavior produces approximate (interval) answers and therefore
/// consumes the per-query error bound. `Plain`/`Agg` are exact; `Sampling`/`A3i`
/// are approximate. This is the single definition of that property.
bool behavior_is_approx(Behavior behavior);

/// A behavior composed with a substrate, with the two substrate-derived flags
/// filled in. The flags are computed here, never hand-set, so a combination
/// like eager materialization over a lazily-cracked substrate is impossible to
/// construct.
struct ResolvedRunConfig {
    EngineConfig        behavior;
    AdaptiveAccessPath* substrate = nullptr;
    /// = substrate->supports_refine(): may the structure be cracked per query.
    bool allow_refine = false;
    /// = persist_summaries && substrate->has_prebuilt_partitions(): may every
    /// partition's summary be precomputed at initialization.
    bool eager_materialize = false;

    static ResolvedRunConfig resolve(EngineConfig behavior,
                                     AdaptiveAccessPath& substrate);
};

}  // namespace a3i

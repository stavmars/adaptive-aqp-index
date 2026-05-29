#include "a3i/engine/method.hpp"

namespace a3i {

EngineConfig behavior_config(Behavior behavior, AllocatorConfig allocator) {
    using AM = EngineConfig::AccuracyMode;
    EngineConfig cfg;
    cfg.allocator = allocator;
    switch (behavior) {
        case Behavior::Plain:
            cfg.accuracy_mode    = AM::ForceExact;
            cfg.persist_summaries = false;
            break;
        case Behavior::Agg:
            cfg.accuracy_mode    = AM::ForceExact;
            cfg.persist_summaries = true;
            break;
        case Behavior::Sampling:
            cfg.accuracy_mode    = AM::PerQuery;
            cfg.persist_summaries = false;
            break;
        case Behavior::A3i:
            cfg.accuracy_mode    = AM::PerQuery;
            cfg.persist_summaries = true;
            break;
    }
    return cfg;
}

std::string run_id(const std::string& substrate_id, Behavior behavior) {
    switch (behavior) {
        case Behavior::Plain:    return substrate_id;
        case Behavior::Agg:      return substrate_id + "_agg";
        case Behavior::Sampling: return substrate_id + "_sampling";
        case Behavior::A3i:      return substrate_id + "_a3i";
    }
    return substrate_id;
}

ResolvedRunConfig ResolvedRunConfig::resolve(EngineConfig behavior,
                                             AdaptiveAccessPath& substrate) {
    ResolvedRunConfig rr;
    rr.behavior          = behavior;
    rr.substrate         = &substrate;
    rr.allow_refine      = substrate.supports_refine();
    rr.eager_materialize = behavior.persist_summaries && substrate.is_fully_built();
    return rr;
}

}  // namespace a3i

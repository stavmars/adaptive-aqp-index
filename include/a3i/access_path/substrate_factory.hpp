// The single place substrates are constructed.
//
// A substrate is created by string id ("adaptive_kd", "static_kd", ...)
// plus a SubstrateConfig. Registration is the only thing a new substrate
// must add; methods and tests obtain substrates exclusively through the
// factory, so any method runs over any registered substrate and the
// substrate-independence test can iterate the whole registry.

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "a3i/access_path/adaptive_access_path.hpp"
#include "a3i/core/geometry.hpp"

namespace a3i {

/// Construction-time parameters shared by substrates. A substrate uses
/// only the fields it needs (the KD substrates use `domain_bounds` for the
/// root and `refinement_threshold` as the crack gate).
struct SubstrateConfig {
    HyperRect     domain_bounds;             ///< Bounds of the root partition.
    std::uint32_t refinement_threshold = 1024;
    bool          stochastic_cracking   = false;
    std::uint32_t leaf_min_size         = 1024;  ///< Used by the static build.
};

class SubstrateFactory {
public:
    using Builder =
        std::function<std::unique_ptr<AdaptiveAccessPath>(const SubstrateConfig&)>;

    /// The process-wide registry.
    static SubstrateFactory& instance();

    /// Register a builder under `id`. Throws std::invalid_argument if `id`
    /// is already registered.
    void register_substrate(const std::string& id, Builder builder);

    /// Construct the substrate registered under `id`. Throws
    /// std::invalid_argument if `id` is unknown.
    std::unique_ptr<AdaptiveAccessPath> create(const std::string& id,
                                               const SubstrateConfig& config) const;

    /// True iff `id` has a registered builder.
    bool is_registered(const std::string& id) const;

    /// All registered ids, sorted, for iterating substrates in tests.
    std::vector<std::string> registered_ids() const;

private:
    std::map<std::string, Builder> builders_;
};

}  // namespace a3i

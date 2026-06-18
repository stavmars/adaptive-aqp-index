#include "a3i/access_path/substrate_factory.hpp"

#include <stdexcept>
#include <utility>

#include "a3i/substrates/adaptive_kd_access_path.hpp"
#include "a3i/substrates/grid_akd_access_path.hpp"
#include "a3i/substrates/static_kd_access_path.hpp"

namespace a3i {

namespace {

// Register every built-in substrate. Called once when the singleton is
// first constructed so the registry is populated before any lookup.
void register_builtins(SubstrateFactory& factory) {
    factory.register_substrate(
        "adaptive_kd",
        [](const SubstrateConfig& config) {
            return std::make_unique<AdaptiveKdAccessPath>(config);
        });
    factory.register_substrate(
        "static_kd",
        [](const SubstrateConfig& config) {
            return std::make_unique<StaticKdAccessPath>(config);
        });
    factory.register_substrate(
        "grid_akd",
        [](const SubstrateConfig& config) {
            return std::make_unique<GridAkdAccessPath>(config);
        });
}

}  // namespace

SubstrateFactory& SubstrateFactory::instance() {
    static SubstrateFactory factory = [] {
        SubstrateFactory f;
        register_builtins(f);
        return f;
    }();
    return factory;
}

void SubstrateFactory::register_substrate(const std::string& id, Builder builder) {
    auto [it, inserted] = builders_.emplace(id, std::move(builder));
    if (!inserted) {
        throw std::invalid_argument("SubstrateFactory: id already registered: " + id);
    }
}

std::unique_ptr<AdaptiveAccessPath> SubstrateFactory::create(
    const std::string& id, const SubstrateConfig& config) const {
    auto it = builders_.find(id);
    if (it == builders_.end()) {
        throw std::invalid_argument("SubstrateFactory: unknown substrate id: " + id);
    }
    return it->second(config);
}

bool SubstrateFactory::is_registered(const std::string& id) const {
    return builders_.find(id) != builders_.end();
}

std::vector<std::string> SubstrateFactory::registered_ids() const {
    std::vector<std::string> ids;
    ids.reserve(builders_.size());
    for (const auto& [id, _] : builders_) ids.push_back(id);
    return ids;  // std::map keeps keys sorted
}

}  // namespace a3i

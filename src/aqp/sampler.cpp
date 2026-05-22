#include "a3i/aqp/sampler.hpp"

#include <random>

namespace a3i {

namespace {

/// Partial Fisher-Yates: move `count` uniformly chosen elements to the front
/// of `eligible` and truncate. Clamps `count` to the eligible size.
std::vector<IndexPos> partial_shuffle(std::vector<IndexPos> eligible,
                                      std::uint64_t count, Rng& rng) {
    const std::uint64_t n = eligible.size();
    if (count > n) count = n;
    for (std::uint64_t i = 0; i < count; ++i) {
        std::uniform_int_distribution<std::uint64_t> dist(i, n - 1);
        const std::uint64_t j = dist(rng);
        std::swap(eligible[static_cast<std::size_t>(i)],
                  eligible[static_cast<std::size_t>(j)]);
    }
    eligible.resize(static_cast<std::size_t>(count));
    return eligible;
}

}  // namespace

std::vector<IndexPos> Sampler::draw_from_range(std::uint64_t size,
                                               const SampleTracker& tracker,
                                               std::uint64_t count, Rng& rng) {
    std::vector<IndexPos> eligible;
    eligible.reserve(static_cast<std::size_t>(size));
    for (std::uint64_t p = 0; p < size; ++p) {
        const auto pos = static_cast<IndexPos>(p);
        if (!tracker.contains(pos)) eligible.push_back(pos);
    }
    return partial_shuffle(std::move(eligible), count, rng);
}

std::vector<IndexPos> Sampler::draw_from_bitset(
    std::span<const IndexPos> candidates, const SampleTracker& tracker,
    std::uint64_t count, Rng& rng) {
    std::vector<IndexPos> eligible;
    eligible.reserve(candidates.size());
    for (IndexPos pos : candidates) {
        if (!tracker.contains(pos)) eligible.push_back(pos);
    }
    return partial_shuffle(std::move(eligible), count, rng);
}

}  // namespace a3i

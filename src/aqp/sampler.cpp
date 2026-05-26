#include "a3i/aqp/sampler.hpp"

#include <random>
#include <unordered_set>

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

/// List the eligible positions (in the universe, not yet in the tracker).
std::vector<IndexPos> enumerate_eligible(EligibleUniverse universe,
                                         const SampleTracker& tracker) {
    std::vector<IndexPos> eligible;
    if (universe.qualifying != nullptr) {
        eligible.reserve(static_cast<std::size_t>(universe.qualifying->count()));
        universe.qualifying->for_each_set([&](IndexPos pos) {
            if (!tracker.contains(pos)) eligible.push_back(pos);
        });
    } else {
        eligible.reserve(static_cast<std::size_t>(universe.size));
        for (std::uint64_t p = 0; p < universe.size; ++p) {
            const auto pos = static_cast<IndexPos>(p);
            if (!tracker.contains(pos)) eligible.push_back(pos);
        }
    }
    return eligible;
}

/// Rejection sampling: propose uniform positions in [0, size) and keep those
/// that qualify, are untracked, and were not already chosen. Allocates only
/// O(count) (the chosen set), never O(size). The caller clamps `count` to the
/// eligible population, so the loop always terminates.
std::vector<IndexPos> reject_sample(EligibleUniverse universe,
                                    const SampleTracker& tracker,
                                    std::uint64_t count, Rng& rng) {
    std::vector<IndexPos> picked;
    picked.reserve(static_cast<std::size_t>(count));
    std::unordered_set<IndexPos> chosen;
    chosen.reserve(static_cast<std::size_t>(count) * 2);
    std::uniform_int_distribution<std::uint64_t> dist(0, universe.size - 1);
    while (picked.size() < count) {
        const auto pos = static_cast<IndexPos>(dist(rng));
        if (universe.qualifying != nullptr &&
            !universe.qualifying->contains(pos)) {
            continue;
        }
        if (tracker.contains(pos)) continue;
        if (!chosen.insert(pos).second) continue;
        picked.push_back(pos);
    }
    return picked;
}

}  // namespace

std::vector<IndexPos> Sampler::draw(EligibleUniverse universe,
                                    const SampleTracker& tracker,
                                    std::uint64_t count, Rng& rng) {
    const std::uint64_t universe_total = universe.qualifying != nullptr
                                             ? universe.qualifying->count()
                                             : universe.size;
    const std::uint64_t taken = tracker.count();
    const std::uint64_t eligible_count =
        universe_total > taken ? universe_total - taken : 0;
    if (count > eligible_count) count = eligible_count;
    if (count == 0) return {};

    // Rejection pays off only when few proposals are wasted: the draw is a
    // small slice of the eligible set, the tracker is sparse, and (for a
    // narrowed universe) the qualifying set is dense in [0, size). Otherwise
    // enumerate, where work is bounded by the eligible population.
    const bool small_draw     = count * 2 <= eligible_count;
    const bool sparse_tracker = taken * 2 <= universe_total;
    const bool dense_qualifying =
        universe.qualifying == nullptr ||
        universe.qualifying->count() * 2 >= universe.size;
    if (small_draw && sparse_tracker && dense_qualifying) {
        return reject_sample(universe, tracker, count, rng);
    }
    return partial_shuffle(enumerate_eligible(universe, tracker), count, rng);
}

}  // namespace a3i

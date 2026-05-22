// Seeded pseudo-random generation for the index.
//
// Randomness uses the standard library's 64-bit Mersenne Twister
// (std::mt19937_64), seeded explicitly so a run is reproducible for a given
// build: the same seed yields the same draws and hence the same result.
// Reproducibility is scoped to one compiler/standard-library toolchain --
// bit-for-bit identical sequences across compilers or library versions are
// not promised, which is acceptable here. Call sites use the standard
// distributions (e.g. std::uniform_int_distribution) directly on the engine;
// there is no custom wrapper.

#pragma once

#include <cstdint>
#include <random>

namespace a3i {

/// The generator type used throughout the index: a plain alias for the
/// standard 64-bit Mersenne Twister. Construct it with an explicit seed and
/// draw through standard distributions at the call site.
using Rng = std::mt19937_64;

/// Derive a reproducible seed from the coordinates of a sampling draw so that
/// cumulative rounds are reproducible and non-overlapping.
inline std::uint64_t mix_seed(std::uint64_t query_ordinal, std::uint64_t round,
                              std::uint64_t stratum_ordinal,
                              std::uint64_t target) {
    std::seed_seq seq{static_cast<std::uint32_t>(query_ordinal),
                      static_cast<std::uint32_t>(query_ordinal >> 32),
                      static_cast<std::uint32_t>(round),
                      static_cast<std::uint32_t>(round >> 32),
                      static_cast<std::uint32_t>(stratum_ordinal),
                      static_cast<std::uint32_t>(stratum_ordinal >> 32),
                      static_cast<std::uint32_t>(target),
                      static_cast<std::uint32_t>(target >> 32)};
    std::uint32_t out[2];
    seq.generate(out, out + 2);
    return (static_cast<std::uint64_t>(out[1]) << 32) | out[0];
}

}  // namespace a3i

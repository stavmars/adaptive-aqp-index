// A compact set of positions drawn from a contiguous range [0, size).
//
// One bit per position, packed into 64-bit words. It answers the two
// questions a without-replacement sampler asks of a candidate set in O(1)
// each -- "is this position a member?" (contains) and "how many members are
// there?" (count) -- while also supporting ascending enumeration of its
// members for the cases that must list them. A running member count is kept
// so count() is O(1) rather than a popcount sweep.
//
// Positions are local to whatever range the caller defines (typically a
// partition's [0, population) positions); the bitset itself only knows the
// size it was created with.

#pragma once

#include <cstdint>
#include <vector>

#include "a3i/core/types.hpp"

namespace a3i {

class PositionBitset {
public:
    explicit PositionBitset(std::uint64_t size)
        : words_((size + 63) / 64, 0), size_(size) {}

    /// Mark `pos` a member. Returns true iff it was not already a member, so
    /// callers can keep their own counts in sync without a second lookup.
    bool set(IndexPos pos) {
        std::uint64_t& w = words_[pos / 64];
        const std::uint64_t bit = std::uint64_t{1} << (pos % 64);
        if (w & bit) return false;
        w |= bit;
        ++count_;
        return true;
    }

    bool contains(IndexPos pos) const {
        return (words_[pos / 64] >> (pos % 64)) & std::uint64_t{1};
    }

    std::uint64_t count() const noexcept { return count_; }
    std::uint64_t size() const noexcept { return size_; }

    /// Invoke `fn(IndexPos)` for every member in ascending order.
    template <typename F>
    void for_each_set(F&& fn) const {
        for (std::size_t wi = 0; wi < words_.size(); ++wi) {
            std::uint64_t w = words_[wi];
            while (w) {
                const unsigned b = __builtin_ctzll(w);
                fn(static_cast<IndexPos>(wi * 64 + b));
                w &= w - 1;  // clear the lowest set bit
            }
        }
    }

    /// Materialize the members as an ascending vector.
    std::vector<IndexPos> to_positions() const {
        std::vector<IndexPos> out;
        out.reserve(static_cast<std::size_t>(count_));
        for_each_set([&](IndexPos p) { out.push_back(p); });
        return out;
    }

private:
    std::vector<std::uint64_t> words_;
    std::uint64_t              count_ = 0;
    std::uint64_t              size_  = 0;
};

}  // namespace a3i

// Sanity check: the build is wired, GoogleTest is reachable, and the
// a3i namespace exposes a non-empty version string.

#include <gtest/gtest.h>

#include "a3i/util/version.hpp"

TEST(Sanity, VersionStringIsNonEmpty) {
    const auto v = a3i::version();
    EXPECT_FALSE(v.empty());
}

TEST(Sanity, ArithmeticHoldsInTheTestFramework) {
    EXPECT_EQ(2 + 2, 4);
}

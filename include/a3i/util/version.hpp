// Project-wide version string.

#pragma once

#include <string_view>

namespace a3i {

constexpr std::string_view kVersion = "0.1.0";

inline std::string_view version() { return kVersion; }

}  // namespace a3i

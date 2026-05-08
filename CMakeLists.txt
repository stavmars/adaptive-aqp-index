# A3I — Accuracy-Aware Adaptive Indexing
# Top-level build. C++20, GoogleTest via FetchContent, CTest enabled.

cmake_minimum_required(VERSION 3.22)
project(a3i LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Default build type" FORCE)
endif()

# Export compile commands for tooling (clangd, IDE).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Third-party dependencies (FetchContent) ----------------------------
include(FetchContent)

FetchContent_Declare(
    fast_float
    GIT_REPOSITORY https://github.com/fastfloat/fast_float.git
    GIT_TAG        v6.1.6
)
FetchContent_MakeAvailable(fast_float)

set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# csv2 — header-only CSV reader with mmap'd input and configurable
# delimiter/quote. Avoids zsv's autoconf build toolchain.
FetchContent_Declare(
    csv2
    GIT_REPOSITORY https://github.com/p-ranav/csv2.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(csv2)

# ---- Core library --------------------------------------------------------
# Static library carrying compiled units shared by apps and tests.
add_library(a3i_core STATIC
    src/storage/index_table.cpp
    src/storage/manifest.cpp
    src/storage/binary_column_store.cpp
    src/tools/csv_to_columns_pipeline.cpp
)
target_include_directories(a3i_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(a3i_core PUBLIC
    fast_float
    nlohmann_json::nlohmann_json
    csv2::csv2
)
target_compile_features(a3i_core PUBLIC cxx_std_20)

# ---- Apps ----------------------------------------------------------------
add_executable(a3i_sanity apps/a3i_sanity.cpp)
target_link_libraries(a3i_sanity PRIVATE a3i_core)

add_executable(convert_csv_to_columns tools/convert_csv_to_columns.cpp)
target_link_libraries(convert_csv_to_columns PRIVATE a3i_core)
set_target_properties(convert_csv_to_columns PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)

# ---- Tests ---------------------------------------------------------------
enable_testing()

set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_subdirectory(tests)

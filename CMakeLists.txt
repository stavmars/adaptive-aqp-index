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

# ---- Core library --------------------------------------------------------
# INTERFACE target centralizing include paths and language settings;
# compiled units are added as the project grows.
add_library(a3i_core INTERFACE)
target_include_directories(a3i_core INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_features(a3i_core INTERFACE cxx_std_20)

# ---- Apps ----------------------------------------------------------------
add_executable(a3i_sanity apps/a3i_sanity.cpp)
target_link_libraries(a3i_sanity PRIVATE a3i_core)

# ---- Tests ---------------------------------------------------------------
enable_testing()

include(FetchContent)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_subdirectory(tests)

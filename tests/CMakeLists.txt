# Each test file becomes its own executable + CTest entry under a
# stable name and label. `unit` is the fast gate; `statistical` is the
# slower seed-sweep gate that certifies the confidence intervals.

include(GoogleTest)

function(a3i_add_labeled_test name label)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE a3i_core GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    gtest_discover_tests(${name}
        TEST_PREFIX  "${name}."
        PROPERTIES LABELS "${label}"
    )
endfunction()

function(a3i_add_test name)
    a3i_add_labeled_test(${name} "unit" ${ARGN})
endfunction()

a3i_add_test(a3i_sanity_test test_sanity.cpp)
a3i_add_test(test_storage   test_storage.cpp)
a3i_add_test(test_binary_column_store test_binary_column_store.cpp)
a3i_add_test(test_geometry test_geometry.cpp)
a3i_add_test(test_exact_scan test_exact_scan.cpp)
a3i_add_test(test_access_path test_access_path.cpp)
a3i_add_test(test_substrate_independence test_substrate_independence.cpp)
a3i_add_test(test_partitioner test_partitioner.cpp)
a3i_add_test(test_static_kd test_static_kd.cpp)
a3i_add_test(test_summary test_summary.cpp)
a3i_add_test(test_partition_state_store test_partition_state_store.cpp)
a3i_add_test(test_sampler test_sampler.cpp)
a3i_add_test(test_stratum_cursor test_stratum_cursor.cpp)
a3i_add_test(test_position_bitset test_position_bitset.cpp)
a3i_add_test(test_decompose test_decompose.cpp)
a3i_add_test(test_tdist test_tdist.cpp)
a3i_add_test(test_estimators test_estimators.cpp)
a3i_add_test(test_allocator test_allocator.cpp)
a3i_add_test(test_query_engine test_query_engine.cpp)
a3i_add_test(test_differential test_differential.cpp)
a3i_add_test(test_methods test_methods.cpp)
a3i_add_test(test_workload_gen test_workload_gen.cpp)
a3i_add_test(test_cell_runner test_cell_runner.cpp)

# Parquet front-end: the CSV/TSV -> Parquet converter and the seeded
# synthetic generator. The per-test entries run in the fast `unit` gate; the
# whole-suite entries below carry the `m_source_parquet` label so the
# data-ingestion path can also be exercised on its own.
a3i_add_test(test_csv_to_parquet test_csv_to_parquet.cpp)
a3i_add_test(test_synthetic_generator test_synthetic_generator.cpp)

add_test(NAME m_source_parquet.csv_to_parquet COMMAND test_csv_to_parquet)
add_test(NAME m_source_parquet.synthetic_generator COMMAND test_synthetic_generator)
set_tests_properties(
    m_source_parquet.csv_to_parquet
    m_source_parquet.synthetic_generator
    PROPERTIES LABELS m_source_parquet
)

# Dataset preparation: a script-level smoke test that drives
# scripts/prepare_dataset.py end to end on tiny fixtures (generate/convert/
# verify, the drop_if+bounds survival rule, null -> NaN, idempotency, and a
# --max-rows subset). Registered only when a Python interpreter is available;
# the C++ build itself never depends on Python. A3I_BUILD_DIR points the script
# at this build's tools/ directory regardless of the build location.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME m_data_prep.prepare_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_data_prep_smoke.py)
    set_tests_properties(m_data_prep.prepare_smoke PROPERTIES
        LABELS m_data_prep
        ENVIRONMENT "A3I_BUILD_DIR=${CMAKE_BINARY_DIR}")
else()
    message(STATUS "Python3 interpreter not found; skipping m_data_prep smoke test")
endif()

# Experiment pipeline: a hermetic end-to-end smoke test driving
# scripts/run_experiments.py and scripts/validate_results.py over a tiny
# synthetic dataset in a temp tree (prepare -> run cells -> validate), asserting
# the exact ladder matches the scan oracle. Warm (no sudo / no cache drop).
if(Python3_Interpreter_FOUND)
    add_test(NAME m_experiments.pipeline_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_experiments_smoke.py)
    set_tests_properties(m_experiments.pipeline_smoke PROPERTIES
        LABELS m_experiments
        ENVIRONMENT "A3I_BUILD_DIR=${CMAKE_BINARY_DIR}")
endif()

# Analysis layer: plotting/load + plotting/aggregate + analyze_results
# diagnostics, on tiny in-process fixtures. Fast and
# deterministic -> the `unit` gate. Skips itself if pandas is unavailable.
if(Python3_Interpreter_FOUND)
    add_test(NAME analysis.test_analysis
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_analysis.py)
    set_tests_properties(analysis.test_analysis PROPERTIES LABELS "unit")
endif()

# Exploratory plotting: the plot_results axis-role inference + ambiguity guard,
# on tiny in-memory frames. Fast and deterministic -> the `unit` gate. Skips
# itself if pandas/matplotlib are unavailable.
if(Python3_Interpreter_FOUND)
    add_test(NAME explore.test_explore
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_explore.py)
    set_tests_properties(explore.test_explore PROPERTIES LABELS "unit")
endif()

# Statistical gate: seeded sanity plus the empirical CI-coverage sweep
# (including the recorded heavy-tail under-coverage). Kept out of the fast
# `unit` label because it runs many seeds.
a3i_add_labeled_test(test_statistical_smoke "statistical" test_statistical_smoke.cpp)
a3i_add_labeled_test(test_statistical_coverage "statistical" test_statistical_coverage.cpp)
a3i_add_labeled_test(test_workload_soundness "statistical" test_workload_soundness.cpp)
# Each test file becomes its own executable + CTest entry under a
# stable name and label (`unit` is the fast gate).

include(GoogleTest)

function(a3i_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE a3i_core GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    gtest_discover_tests(${name}
        TEST_PREFIX  "${name}."
        PROPERTIES LABELS "unit"
    )
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
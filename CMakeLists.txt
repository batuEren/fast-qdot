cmake_minimum_required (VERSION 3.14)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("fast-qdot")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Main executable ────────────────────────────────────────────────────────────
add_executable (fast-qdot "fast-qdot.cpp" "fast-qdot.h" "src/naive.cpp" "src/lut.cpp" "src/lut.h")

# ── Dependencies via FetchContent ──────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
  GIT_SHALLOW    TRUE
)

# prevents overriding
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.0
  GIT_SHALLOW    TRUE
)
set(BENCHMARK_ENABLE_TESTING   OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# ── Tests ──────────────────────────────────────────────────────────────────────
enable_testing()

if(MSVC)
    set_source_files_properties(src/mad_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
else()
    set_source_files_properties(src/mad_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
endif()

add_executable(fast-qdot-tests
    tests/test_main.cpp
    tests/test_mad.cpp
    tests/test_mad_avx2.cpp
    "src/naive.cpp"
    "src/mad.cpp"
    "src/mad_avx2.cpp"
 "src/lut.cpp" "src/lut.h")
target_link_libraries(fast-qdot-tests PRIVATE GTest::gtest_main)
target_include_directories(fast-qdot-tests PRIVATE ${CMAKE_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(fast-qdot-tests)

# ── Benchmarks ─────────────────────────────────────────────────────────────────
add_executable(fast-qdot-bench benchmarks/bench_main.cpp "src/naive.cpp" "src/mad.cpp" "src/mad_avx2.cpp" "src/lut.cpp" "src/lut.h")
target_link_libraries(fast-qdot-bench PRIVATE benchmark::benchmark_main)
target_include_directories(fast-qdot-bench PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(fast-qdot-bench PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -march=native>
)

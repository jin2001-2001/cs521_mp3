cmake_minimum_required(VERSION 3.14)
project(DualNumberTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# Prevent GTest from overriding main()
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add the test target
add_executable(run_tests dual_test.cpp)
target_link_libraries(run_tests gtest_main)

# Tell CTest to run this binary
include(GoogleTest)
gtest_discover_tests(run_tests)

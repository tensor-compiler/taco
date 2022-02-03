file(GLOB GTEST_SOURCE gtest/gtest-all.cc)
file(GLOB GTEST_HEADER gtest/gtest.h)
include_directories(${TACO_TEST_DIR} ${TACO_SRC_DIR})
add_library(taco-gtest ${GTEST_HEADER} ${GTEST_SOURCE})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-declarations -DTACO_TEST_DIR=\"${TACO_TEST_DIR}\"")

file(GLOB TEST_HEADERS *.h)
file(GLOB TEST_SOURCES *.cpp)

add_executable(taco-test ${TEST_SOURCES} ${TEST_HEADERS})
target_link_libraries(taco-test taco-gtest)
target_link_libraries(taco-test pthread)
target_link_libraries(taco-test taco)

if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
  add_test(NAME taco-test COMMAND taco-test)
else()
  cmake_policy(SET CMP0054 NEW)
  cmake_policy(SET CMP0057 NEW)
  include(GoogleTest)
  gtest_add_tests(TARGET taco-test)
endif()

# This will run all *.bats test files in the source test/ folder.
add_test(NAME taco-cli-test COMMAND ${CMAKE_SOURCE_DIR}/test/bats/bin/bats ${CMAKE_SOURCE_DIR}/test/)

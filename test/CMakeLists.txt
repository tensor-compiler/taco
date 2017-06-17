file(GLOB GTEST_SOURCE gtest/gtest-all.cc)
file(GLOB GTEST_HEADER gtest/gtest.h)
include_directories(${TACO_TEST_DIR} ${TACO_SRC_DIR})
add_library(gtest ${GTEST_HEADER} ${GTEST_SOURCE})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-declarations -DTACO_TEST_DIR=\"${TACO_TEST_DIR}\"")

file(GLOB TEST_HEADERS *.h)
file(GLOB TEST_SOURCES *.cpp)

add_executable(taco-test ${TEST_SOURCES} ${TEST_HEADERS})
target_link_libraries(taco-test gtest)
target_link_libraries(taco-test pthread)
target_link_libraries(taco-test taco)

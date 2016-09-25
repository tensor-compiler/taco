file(GLOB GTEST_SOURCE gtest/gtest-all.cc)
file(GLOB GTEST_HEADER gtest/gtest.h)
include_directories(${TAC_TEST_DIR})
add_library(gtest ${GTEST_HEADER} ${GTEST_SOURCE})

file(GLOB TEST_HEADERS *.h)
file(GLOB TEST_SOURCES *.cpp)

add_executable(tac-test ${TEST_SOURCES} ${TEST_HEADERS})
target_link_libraries(tac-test gtest)
target_link_libraries(tac-test pthread)
target_link_libraries(tac-test tac)

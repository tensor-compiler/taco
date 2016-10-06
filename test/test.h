#ifndef TEST_H
#define TEST_H

#include "gtest/gtest.h"
#include <iostream>

namespace taco {
namespace test {
}
}
using namespace taco;
using namespace taco::test;

using namespace std;

using ::testing::TestWithParam;
using ::testing::tuple;
using ::testing::Bool;
using ::testing::Values;
using ::testing::Combine;


#endif

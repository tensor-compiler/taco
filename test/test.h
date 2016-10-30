#ifndef TEST_H
#define TEST_H

#include "gtest/gtest.h"

#include <iostream>
#include <vector>
#include <memory>

#include "util/strings.h"

namespace taco { namespace test {} }

using namespace taco::test;
using namespace std;

namespace taco {
template <typename T> class Tensor;
class Var;
class Expr;

namespace test {

using ::testing::TestWithParam;
using ::testing::tuple;
using ::testing::Bool;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::Combine;

template <typename T>
void ASSERT_ARRAY_EQ(vector<T> expected, std::pair<T*,size_t> actual) {
  SCOPED_TRACE(string("expected: ") + "{" + util::join(expected) + "}");
  SCOPED_TRACE(string("  actual: ") + "{"
      + util::join(&actual.first[0], &actual.first[actual.second])
      + "}");

  ASSERT_EQ(expected.size(), actual.second);
  for (size_t i=0; i < expected.size(); ++i) {
    ASSERT_FLOAT_EQ(expected[i], actual.first[i]);
  }
}

template <typename T>
void ASSERT_VECTOR_EQ(std::vector<T> expected,
                      std::vector<T> actual) {
  SCOPED_TRACE(string("expected: ") + "{" + util::join(expected) + "}");
  SCOPED_TRACE(string("  actual: ") + "{" + util::join(actual) + "}");
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t k=0; k < actual.size(); ++k) {
    ASSERT_EQ(expected[k], actual[k]);
  }
}

}}

#endif

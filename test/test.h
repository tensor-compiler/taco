#ifndef TEST_H
#define TEST_H

#include "gtest/gtest.h"

#include <iostream>
#include <vector>
#include <memory>

#include "taco/format.h"
#include "taco/tensor.h"
#include "taco/util/error.h"
#include "taco/util/strings.h"

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

std::string testDirectory();
std::string testDataDirectory();

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

template <typename T>
void ASSERT_TENSOR_EQ(Tensor<T> expected,
                      Tensor<T> actual) {
  SCOPED_TRACE(string("expected: ") + util::toString(expected) );
  SCOPED_TRACE(string("  actual: ") + util::toString(actual) );
  ASSERT_TRUE(equals(expected, actual));
}

template <typename T>
void ASSERT_STORAGE_EQUALS(vector<vector<vector<int>>> expectedIndices,
                           vector<T> expectedValues,
                           Tensor<T> actual) {
  auto storage = actual.getStorage();
  auto levels = storage.getFormat().getLevels();

  // Check that the indices are as expected
  auto size = storage.getSize();

  for (size_t i=0; i < levels.size(); ++i) {
    auto expectedIndex = expectedIndices[i];
    auto index = storage.getDimensionIndex(i);

    switch (levels[i].getType()) {
      case LevelType::Dense: {
        taco_iassert(expectedIndex.size() == 1) <<
            "Dense indices have a ptr array";
        ASSERT_EQ(1u, index.size());
        ASSERT_ARRAY_EQ(expectedIndex[0], {index[0], size.numIndexValues(i,0)});
        break;
      }
      case LevelType::Sparse:
      case LevelType::Fixed: {
        taco_iassert(expectedIndex.size() == 2);
        ASSERT_EQ(2u, index.size());
        ASSERT_ARRAY_EQ(expectedIndex[0], {index[0], size.numIndexValues(i,0)});
        ASSERT_ARRAY_EQ(expectedIndex[1], {index[1], size.numIndexValues(i,1)});
        break;
      }
    }
  }

  ASSERT_EQ(expectedValues.size(), storage.getSize().numValues());
  ASSERT_ARRAY_EQ(expectedValues, {storage.getValues(), size.numValues()});
}

}}

#endif

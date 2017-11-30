#ifndef TEST_H
#define TEST_H

#include "gtest/gtest.h"

#include <iostream>
#include <vector>
#include <memory>

#include "taco/format.h"
#include "taco/tensor.h"
#include "taco/error.h"
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
void ASSERT_ARRAY_EQ(vector<T> expected, std::pair<const T*,size_t> actual) {
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

void ASSERT_TENSOR_EQ(const TensorBase& expected, const TensorBase& actual);

template <typename T>
void ASSERT_STORAGE_EQUALS(vector<vector<vector<int>>> expectedIndices,
                           vector<T> expectedValues,
                           Tensor<T> actual) {
  auto storage = actual.getStorage();

  auto index = storage.getIndex();
  for (size_t i=0; i < storage.getFormat().getOrder(); ++i) {
    auto modeIndex = index.getModeIndex(i);
    switch (storage.getFormat().getModeTypes()[i]) {
      case ModeType::Dense: {
        taco_iassert(expectedIndices[i].size() == 1);
        ASSERT_EQ(1u, modeIndex.numIndexArrays());
        auto size = modeIndex.getIndexArray(0);
        ASSERT_ARRAY_EQ(expectedIndices[i][0],
                        {(int*)size.getData(), size.getSize()});
        break;
      }
      case ModeType::Sparse:
      case ModeType::Fixed: {
        taco_iassert(expectedIndices[i].size() == 2);
        ASSERT_EQ(2u, modeIndex.numIndexArrays());
        auto pos = modeIndex.getIndexArray(0);
        auto idx = modeIndex.getIndexArray(1);
        ASSERT_ARRAY_EQ(expectedIndices[i][0],
                        {(int*)pos.getData(), pos.getSize()});
        ASSERT_ARRAY_EQ(expectedIndices[i][1],
                        {(int*)idx.getData(), idx.getSize()});
        break;
      }
    }
  }

  auto nnz = index.getSize();
  ASSERT_EQ(expectedValues.size(), nnz);
  ASSERT_ARRAY_EQ(expectedValues, {(double*)storage.getValues().getData(),nnz});
}

}}

#endif

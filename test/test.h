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

void ASSERT_STORAGE_EQ(TensorStorage expected, TensorStorage actual);
void ASSERT_TENSOR_EQ(TensorBase expected, TensorBase actual);

template <typename T>
void ASSERT_COMPONENTS_EQUALS(vector<vector<vector<int>>> expectedIndices,
                              vector<T> expectedValues, Tensor<T> actual) {
  auto storage = actual.getStorage();

  auto index = storage.getIndex();
  for (int i=0; i < storage.getFormat().getOrder(); ++i) {
    auto modeIndex = index.getModeIndex(i);
    auto modeType = storage.getFormat().getModeFormats()[i];
    if (modeType == ModeFormat::Dense) {
      taco_iassert(expectedIndices[i].size() == 1);
      ASSERT_EQ(1, modeIndex.numIndexArrays());
      auto size = modeIndex.getIndexArray(0);
      ASSERT_ARRAY_EQ(expectedIndices[i][0],
                      {(int*)size.getData(), size.getSize()});
    } else if (modeType == ModeFormat::Sparse) {
      taco_iassert(expectedIndices[i].size() == 2);
      ASSERT_EQ(2, modeIndex.numIndexArrays());
      auto pos = modeIndex.getIndexArray(0);
      auto idx = modeIndex.getIndexArray(1);
      ASSERT_ARRAY_EQ(expectedIndices[i][0],
                      {(int*)pos.getData(), pos.getSize()});
      ASSERT_ARRAY_EQ(expectedIndices[i][1],
                      {(int*)idx.getData(), idx.getSize()});
    }
  }

  auto nnz = index.getSize();
  ASSERT_EQ(expectedValues.size(), nnz);
  ASSERT_ARRAY_EQ(expectedValues, {(double*)storage.getValues().getData(),nnz});
}

struct NotationTest {
  NotationTest(IndexStmt actual, IndexStmt expected)
      : actual(actual), expected(expected) {}
  IndexStmt actual;
  IndexStmt expected;
};
ostream& operator<<(ostream&, const NotationTest&);


#define ASSERT_NOTATION_EQ(expected, actual)                   \
do {                                                           \
  ASSERT_EQ(util::toString(expected), util::toString(actual)); \
} while (0)

}}

#endif

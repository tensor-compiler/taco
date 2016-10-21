#ifndef TEST_H
#define TEST_H

#include "gtest/gtest.h"

#include <iostream>
#include <vector>

#include "util/strings.h"

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

template <typename T>
void ASSERT_ARRAY_EQ(const T* actual, vector<T> expected) {
  for (size_t i=0; i < expected.size(); ++i) {
    ASSERT_FLOAT_EQ(expected[i], ((T*)actual)[i]);
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

// Class used with parameterized testing. Stores a tensor and the expected
// indices and values from running the test.
struct TestData {
  TestData(Tensor<double> tensor,
           const PackedTensor::Indices& expectedIndices,
           const vector<double> expectedValues)
      : tensor(tensor),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
  }
  Tensor<double> tensor;

  // Expected values
  PackedTensor::Indices expectedIndices;
  vector<double> expectedValues;
};

ostream &operator<<(ostream& os, const TestData& data) {
  os << util::join(data.tensor.getDimensions(), "x")
     << " (" << data.tensor.getFormat() << ")";
  return os;
}

#endif

#include "test.h"

#include <iostream>
#include <map>

#include "tensor.h"
#include "format.h"
#include "packed_tensor.h"
#include "util/strings.h"

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

struct TensorData {
  TensorData(vector<size_t> dimensions, string format,
             vector<pair<vector<int>,double>> coords,
             size_t expectedEmptyNnz, size_t expectedNnz)
      : dimensions(dimensions), format(format), coords(coords),
        expectedEmptyNnz(expectedEmptyNnz), expectedNnz(expectedNnz) {
  }
  vector<size_t> dimensions;
  Format format;
  vector<pair<vector<int>,double>> coords;

  // Expected values
  size_t expectedEmptyNnz;
  size_t expectedNnz;

  Tensor<double> getTensor() const {
    return Tensor<double>(dimensions, format);
  }
};

ostream &operator<<(ostream& os, const TensorData& data) {
  os << util::join(data.dimensions, "x") << " (" << data.format << ")";
  return os;
}

struct tensor : public TestWithParam<TensorData> {
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_P(tensor, empty) {
  Tensor<double> tensor = GetParam().getTensor();
  ASSERT_EQ(GetParam().dimensions.size(), tensor.getOrder());
  tensor.pack();
  ASSERT_EQ(GetParam().expectedEmptyNnz, tensor.getPackedTensor()->getNnz());
}

TEST_P(tensor, pack) {
  Tensor<double> tensor = GetParam().getTensor();
  for (auto& coord : GetParam().coords) {
    tensor.insert(coord.first, coord.second);
  }
  tensor.pack();

  auto tensorPack = tensor.getPackedTensor();
  ASSERT_EQ(GetParam().expectedNnz, tensorPack->getNnz());
}

INSTANTIATE_TEST_CASE_P(storage, tensor,
                        Values(TensorData({1}, "d",
                                          {{{0},1.0}},
                                          1, 1),
                               TensorData({5}, "d",
                                          {{{4},2.0},
                                           {{1},1.0}},
                                          5, 5),
                               TensorData({1}, "s",
                                          {{{0},1.0}},
                                          0, 1),
                               TensorData({5}, "s",
                                          {{{4},2.0},
                                           {{1},1.0}},
                                          0, 2))
                        );


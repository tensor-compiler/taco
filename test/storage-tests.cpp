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
             const vector<pair<vector<int>,double>>& coords,
             const PackedTensor::Indices& expectedIndices)
      : dimensions(dimensions), format(format), coords(coords),
        expectedIndices(expectedIndices) {
  }
  vector<size_t> dimensions;
  Format format;
  vector<pair<vector<int>,double>> coords;

  // Expected values
  PackedTensor::Indices expectedIndices;

  Tensor<double> getTensor() const {
    return Tensor<double>(dimensions, format);
  }
};

ostream &operator<<(ostream& os, const TensorData& data) {
  os << util::join(data.dimensions, "x") << " (" << data.format << ")";
  return os;
}

struct storage : public TestWithParam<TensorData> {
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_P(storage, pack) {
  Tensor<double> tensor = GetParam().getTensor();
  for (auto& coord : GetParam().coords) {
    tensor.insert(coord.first, coord.second);
  }
  tensor.pack();

  auto tensorPack = tensor.getPackedTensor();
//  ASSERT_EQ(GetParam().expectedNnz, tensorPack->getNnz());

  // Check that the indices are as expected
  auto expectedIndices = GetParam().expectedIndices;
  auto         indices = tensorPack->getIndices();
  ASSERT_EQ(expectedIndices.size(), indices.size());

  for (size_t i=0; i < indices.size(); ++i) {
    auto expectedIndex = expectedIndices[i];
    auto         index = indices[i];
    ASSERT_EQ(expectedIndex.size(), index.size());
    for (size_t j=0; j < index.size(); ++j) {
      auto expectedIndexArray = expectedIndex[j];
      auto         indexArray = index[j];
      ASSERT_EQ(expectedIndexArray.size(), indexArray.size());
      for (size_t k=0; k < indexArray.size(); ++k) {
        SCOPED_TRACE(string("expectedIndexArray: ") + "{" +
                     util::join(expectedIndexArray) + "}");
        SCOPED_TRACE(string("indexArray: ") + "{" +
                     util::join(indexArray) + "}");
        ASSERT_EQ(expectedIndexArray[k], indexArray[k]);
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(vector, storage,
                        Values(TensorData({1}, "d",
                                          {{{0}, 1.0}},
                                          {
                                            {
                                              // Dense index
                                            }
                                          }
                                         ),
                               TensorData({5}, "d",
                                          {
                                            {{4}, 2.0},
                                            {{1}, 1.0},
                                          },
                                          {
                                            {
                                              // Dense index
                                            }
                                          }
                                         ),
                               TensorData({1}, "s",
                                          {{{0}, 1.0}},
                                          {
                                            {
                                              // Sparse index
                                              {0,1},
                                              {0}
                                            }
                                          }
                                         ),
                               TensorData({5}, "s",
                                          {
                                            {{4}, 2.0},
                                            {{1}, 1.0},
                                          },
                                          {
                                            {
                                              // Sparse index
                                              {0,2},
                                              {1,4}
                                            },
                                          }
                                         )
                              )
                        );

INSTANTIATE_TEST_CASE_P(matrix, storage,
                        Values(TensorData({3,3}, "dd",
                                          {
                                            {{0,1}, 1.0},
                                            {{2,2}, 3.0},
                                            {{2,0}, 2.0},
                                          },
                                          {
                                            {
                                              // Dense index
                                            },
                                            {
                                              // Dense index
                                            }
                                          }
                                         ),
                               TensorData({3,3}, "ds",  // CSR
                                          {
                                            {{0,1}, 1.0},
                                            {{2,2}, 3.0},
                                            {{2,0}, 2.0},
                                          },
                                          {
                                            {
                                              // Dense index
                                            },
                                            {
                                              // Sparse index
                                              {0, 1, 1, 3},
                                              {1, 0, 2},
                                            }
                                          }
                                         )
                              )
                        );


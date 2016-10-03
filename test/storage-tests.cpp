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
             const PackedTensor::Indices& expectedIndices,
             const vector<double> expectedValues)
      : dimensions(dimensions), format(format), coords(coords),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
  }
  vector<size_t> dimensions;
  Format format;
  vector<pair<vector<int>,double>> coords;

  // Expected values
  PackedTensor::Indices expectedIndices;
  vector<double> expectedValues;

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

  // Check that the indices are as expected
  auto& expectedIndices = GetParam().expectedIndices;
  auto&         indices = tensorPack->getIndices();
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
        SCOPED_TRACE(string("        indexArray: ") + "{" +
                     util::join(indexArray) + "}");
        ASSERT_EQ(expectedIndexArray[k], indexArray[k]);
      }
    }
  }

  auto& expectedValues = GetParam().expectedValues;
//  ASSERT_EQ(expectedValues.size(), tensorPack->getNnz());
  auto values = tensorPack->getValues();
  for (size_t i=0; i < values.size(); ++i) {
    SCOPED_TRACE(string("expectedValues: ") + "{" +
                 util::join(expectedValues) + "}");
    SCOPED_TRACE(string("        values: ") + "{" +
                 util::join(values) + "}");
//    ASSERT_FLOAT_EQ(expectedValues[i], values[i]);
  }
}

INSTANTIATE_TEST_CASE_P(vector, storage,
                        Values(TensorData({1}, "d",
                                          {
                                            {{0}, 1}
                                          },
                                          {
                                            {
                                              // Dense index
                                            }
                                          },
                                          {1}
                                         ),
                               TensorData({5}, "d",
                                          {
                                            {{4}, 2},
                                            {{1}, 1},
                                          },
                                          {
                                            {
                                              // Dense index
                                            }
                                          },
                                          {0, 1, 0, 0, 2}
                                         ),
                               TensorData({1}, "s",
                                          {
                                            {{0}, 1}
                                          },
                                          {
                                            {
                                              // Sparse index
                                              {0,1},
                                              {0}
                                            }
                                          },
                                          {1}
                                         ),
                               TensorData({5}, "s",
                                          {
                                            {{4}, 2},
                                            {{1}, 1},
                                          },
                                          {
                                            {
                                              // Sparse index
                                              {0,2},
                                              {1,4}
                                            },
                                          },
                                          {1, 2}
                                         )
                              )
                        );

INSTANTIATE_TEST_CASE_P(matrix, storage,
                        Values(TensorData({3,3}, "dd",
                                          {
                                            {{0,1}, 1},
                                            {{2,2}, 3},
                                            {{2,0}, 2},
                                          },
                                          {
                                            {
                                              // Dense index
                                            },
                                            {
                                              // Dense index
                                            }
                                          },
                                          {0, 1, 0,
                                           0, 0, 0,
                                           2, 0, 3}
                                         ),
                               TensorData({3,3}, "ds",  // CSR
                                          {
                                            {{0,1}, 1},
                                            {{2,2}, 3},
                                            {{2,0}, 2},
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
                                          },
                                          {1, 2, 3}
                                         ),
                               TensorData({3,3}, "sd",  // Blocked sparse vec
                                          {
                                            {{0,1}, 1},
                                            {{2,2}, 3},
                                            {{2,0}, 2},
                                          },
                                          {
                                            {
                                              // Sparse index
                                              {0, 2},
                                              {0, 2},
                                            },
                                            {
                                              // Dense index
                                            }
                                          },
                                          {1, 2, 3}
                                         ),
                               TensorData({3,3}, "ss",  // DCSR
                                          {
                                            {{0,1}, 1},
                                            {{2,2}, 3},
                                            {{2,0}, 2},
                                          },
                                          {
                                            {
                                              // Sparse index
                                              {0, 2},
                                              {0, 2},
                                            },
                                            {
                                              // Sparse index
                                              {0, 1, 3},
                                              {1, 0, 2},
                                            }
                                          },
                                          {1, 2, 3}
                                         )
                              )
                        );

INSTANTIATE_TEST_CASE_P(tensor3, storage,
                        Values(TensorData({2,3,3}, "ddd",
                                          {
                                            {{0,0,0}, 1},
                                            {{0,0,1}, 2},
                                            {{0,2,2}, 3},
                                            {{1,0,1}, 4},
                                            {{1,2,0}, 5},
                                            {{1,2,2}, 6}
                                          },
                                          {
                                            {
                                              // Dense index
                                            },
                                            {
                                              // Dense index
                                            },
                                            {
                                              // Dense index
                                            }
                                          },
                                          {1, 2, 0,
                                           0, 0, 0,
                                           0, 0, 3,

                                           0, 4, 0,
                                           0, 0, 0,
                                           5, 0, 6}
                                         ),
                               TensorData({2,3,3}, "sdd",
                                          {
                                            {{0,0,0}, 1},
                                            {{0,0,1}, 2},
                                            {{0,2,2}, 3},
                                            {{1,0,1}, 4},
                                            {{1,2,0}, 5},
                                            {{1,2,2}, 6}
                                          },
                                          {
                                            {
                                              // Sparse index
                                              {0,2},
                                              {0,1}
                                            },
                                            {
                                              // Dense index
                                            } ,
                                            {
                                              // Dense index
                                            }
                                          },
                                          {1, 2, 0,
                                           0, 0, 0,
                                           0, 0, 3,

                                           0, 4, 0,
                                           0, 0, 0,
                                           5, 0, 6}
                                         ),
                               TensorData({2,3,3}, "dsd",
                                          {
                                            {{0,0,0}, 1},
                                            {{0,0,1}, 2},
                                            {{0,2,2}, 3},
                                            {{1,0,1}, 4},
                                            {{1,2,0}, 5},
                                            {{1,2,2}, 6}
                                          },
                                          {
                                            {
                                              // Dense index
                                            },
                                            {
                                              // Sparse index
                                              {0,2,4},
                                              {0,2,0,2}
                                            },
                                            {
                                              // Dense index
                                            }
                                          },
                                          {1, 2, 0,
                                           0, 0, 3,

                                           0, 4, 0,
                                           5, 0, 6}
                                         ),
                      TensorData({2,3,3}, "ssd",
                                          {
                                            {{0,0,0}, 1},
                                            {{0,0,1}, 2},
                                            {{0,2,2}, 3},
                                            {{1,0,1}, 4},
                                            {{1,2,0}, 5},
                                            {{1,2,2}, 6}
                                          },
                                          {
                                            {
                                              // Sparse index
                                              {0,2},
                                              {0,1}
                                            },
                                            {
                                              // Sparse index
                                              {0,2,4},
                                              {0,2,0,2}
                                            },
                                            {
                                              // Dense index
                                            }
                                          },
                                          {1, 2, 0,
                                           0, 0, 3,

                                           0, 4, 0,
                                           5, 0, 6}
                                         ),
                      TensorData({2,3,3}, "dds",
                                          {
                                            {{0,0,0}, 1},
                                            {{0,0,1}, 2},
                                            {{0,2,2}, 3},
                                            {{1,0,1}, 4},
                                            {{1,2,0}, 5},
                                            {{1,2,2}, 6}
                                          },
                                          {
                                            {
                                              // Dense index
                                            },
                                            {
                                              // Dense index
                                            },
                                            {
                                              // Sparse index
                                              {0,2,2,3,4,4,6},
                                              {0,1,2, 1,0,2}
                                            }
                                          },
                                          {1, 2, 3, 4, 5, 6}
                                         ),
                      TensorData({2,3,3}, "sds",
                                          {
                                            {{0,0,0}, 1},
                                            {{0,0,1}, 2},
                                            {{0,2,2}, 3},
                                            {{1,0,1}, 4},
                                            {{1,2,0}, 5},
                                            {{1,2,2}, 6}
                                          },
                                          {
                                            {
                                              // Sparse index
                                              {0,2},
                                              {0,1}
                                            },
                                            {
                                              // Dense index
                                            },
                                            {
                                              // Sparse index
                                              {0,2,2,3,4,4,6},
                                              {0,1,2, 1,0,2}
                                            }
                                          },
                                          {1, 2, 3, 4, 5, 6}
                                         ),
                      TensorData({2,3,3}, "dss",
                                          {
                                            {{0,0,0}, 1},
                                            {{0,0,1}, 2},
                                            {{0,2,2}, 3},
                                            {{1,0,1}, 4},
                                            {{1,2,0}, 5},
                                            {{1,2,2}, 6}
                                          },
                                          {
                                            {
                                              // Dense index
                                            },
                                            {
                                              // Sparse index
                                              {0,2,4},
                                              {0,2,0,2}
                                            },
                                            {
                                              // Sparse index
                                              {0,2,3,4,6},
                                              {0,1,2, 1,0,2}
                                            }
                                          },
                                          {1, 2, 3, 4, 5, 6}
                                         ),
                      TensorData({2,3,3}, "sss",
                                          {
                                            {{0,0,0}, 1},
                                            {{0,0,1}, 2},
                                            {{0,2,2}, 3},
                                            {{1,0,1}, 4},
                                            {{1,2,0}, 5},
                                            {{1,2,2}, 6}
                                          },
                                          {
                                            {
                                              // Sparse index
                                              {0,2},
                                              {0,1}
                                            },
                                            {
                                              // Sparse index
                                              {0,2,4},
                                              {0,2,0,2}
                                            },
                                            {
                                              // Sparse index
                                              {0,2,3,4,6},
                                              {0,1,2, 1,0,2}
                                            }
                                          },
                                          {1, 2, 3, 4, 5, 6}
                                         )
                              )
                        );

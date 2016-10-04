#include "test.h"
#include "test_tensors.h"

#include <map>

#include "tensor.h"
#include "format.h"
#include "packed_tensor.h"
#include "util/strings.h"

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

struct TestData {
  TestData(TensorData tensorData,
           string format,
           const PackedTensor::Indices& expectedIndices,
           const vector<double> expectedValues)
      : tensorData(tensorData), format(format),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
  }
  TensorData tensorData;
  Format format;

  // Expected values
  PackedTensor::Indices expectedIndices;
  vector<double> expectedValues;

  Tensor<double> getTensor() const {
    return Tensor<double>(tensorData.dimensions, format);
  }
};

ostream &operator<<(ostream& os, const TestData& data) {
  os << util::join(data.tensorData.dimensions, "x")
     << " (" << data.format << ")";
  return os;
}

struct storage : public TestWithParam<TestData> {
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_P(storage, pack) {
  TensorData tensorData = GetParam().tensorData;
  Tensor<double> tensor = GetParam().getTensor();

  for (auto& tensorValue : tensorData.values) {
    tensor.insert(tensorValue.coord, tensorValue.value);
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
  ASSERT_EQ(expectedValues.size(), tensorPack->getNnz());
  auto values = tensorPack->getValues();
  for (size_t i=0; i < values.size(); ++i) {
    SCOPED_TRACE(string("expectedValues: ") + "{" +
                 util::join(expectedValues) + "}");
    SCOPED_TRACE(string("        values: ") + "{" +
                 util::join(values) + "}");
    ASSERT_FLOAT_EQ(expectedValues[i], values[i]);
  }
}

INSTANTIATE_TEST_CASE_P(vector, storage,
                        Values(TestData(vector1a,
                                        "d",
                                        {
                                          {
                                            // Dense index
                                          }
                                        },
                                        {1}
                                        ),
                               TestData(vector1a,
                                        "s",
                                        {
                                          {
                                            // Sparse index
                                            {0,1},
                                            {0}
                                          }
                                        },
                                        {1}
                                        ),
                               TestData(vector5a,
                                        "d",
                                        {
                                          {
                                            // Dense index
                                          }
                                        },
                                        {0, 1, 0, 0, 2}
                                        ),
                               TestData(vector5a,
                                        "s",
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
                        Values(TestData(matrix33a,
                                        "dd",
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
                               TestData(matrix33a,
                                        "sd",  // Blocked sparse vec
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
                                        {0, 1, 0,
                                          2, 0, 3}
                                        ),
                               TestData(matrix33a,
                                        "ds",  // CSR
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
                               TestData(matrix33a,
                                        "ss",  // DCSR
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
                        Values(TestData(tensor233a,
                                        "ddd",
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
                               TestData(tensor233a,
                                        "sdd",
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
                               TestData(tensor233a,
                                        "dsd",
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
                               TestData(tensor233a,
                                        "ssd",
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
                               TestData(tensor233a,
                                        "dds",
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
                               TestData(tensor233a,
                                        "sds",
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
                               TestData(tensor233a,
                                        "dss",
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
                               TestData(tensor233a,
                                        "sss",
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

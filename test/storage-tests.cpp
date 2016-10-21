#include "test.h"
#include "test_tensors.h"

#include <map>

#include "tensor.h"
#include "format.h"
#include "packed_tensor.h"
#include "util/strings.h"

struct storage : public TestWithParam<TestData> {
};

TEST_P(storage, pack) {
  Tensor<double> tensor = GetParam().tensor;

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
      ASSERT_VECTOR_EQ(expectedIndex[j], index[j]);
    }
  }

  auto& expectedValues = GetParam().expectedValues;
  ASSERT_EQ(expectedValues.size(), tensorPack->getNnz());
  auto values = tensorPack->getValues();
  ASSERT_VECTOR_EQ(expectedValues, values);
}

INSTANTIATE_TEST_CASE_P(vector, storage,
                        Values(TestData(vectord1a("d"),
                                        {
                                          {
                                            // Dense index
                                          }
                                        },
                                        {1}
                                        ),
                               TestData(vectord1a("s"),
                                        {
                                          {
                                            // Sparse index
                                            {0,1},
                                            {0}
                                          }
                                        },
                                        {1}
                                        ),
                               TestData(vectord5a("d"),
                                        {
                                          {
                                            // Dense index
                                          }
                                        },
                                        {0, 1, 0, 0, 2}
                                        ),
                               TestData(vectord5a("s"),
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
                        Values(TestData(matrixd33a("dd"),
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
                               TestData(matrixd33a("sd"),  // Blocked sparse vec
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
                               TestData(matrixd33a("ds"),  // CSR
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
                               TestData(matrixd33a("ss"),  // DCSR
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
                        Values(TestData(tensord233a("ddd"),
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
                               TestData(tensord233a("sdd"),
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
                               TestData(tensord233a("dsd"),
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
                               TestData(tensord233a("ssd"),
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
                               TestData(tensord233a("dds"),
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
                               TestData(tensord233a("sds"),
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
                               TestData(tensord233a("dss"),
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
                               TestData(tensord233a("sss"),
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

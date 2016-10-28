#include "test.h"
#include "test_tensors.h"

#include <map>

#include "tensor.h"
#include "expr.h"
#include "format.h"
#include "packed_tensor.h"
#include "util/strings.h"

using namespace taco;

struct TestData {
  TestData(Tensor<double> tensor,
           const PackedTensor::Indices& expectedIndices,
           const vector<double> expectedValues)
      : tensor(tensor),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
  }

  Tensor<double>        tensor;
  PackedTensor::Indices expectedIndices;
  vector<double>        expectedValues;
};

static ostream &operator<<(ostream& os, const TestData& data) {
  os << util::join(data.tensor.getDimensions(), "x")
     << " (" << data.tensor.getFormat() << ")";
  return os;
}

struct storage : public TestWithParam<TestData> {};

TEST_P(storage, pack) {
  Tensor<double> tensor = GetParam().tensor;

  auto tensorPack = tensor.getPackedTensor();
  ASSERT_NE(nullptr, tensorPack);
  
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
  ASSERT_ARRAY_EQ(values, expectedValues);
}

INSTANTIATE_TEST_CASE_P(vector, storage,
    Values(TestData(d1a("a", Format({Dense})),
                    {
                      {
                        // Dense index
                      }
                    },
                    {1}
                    ),
           TestData(d1a("a", Format({Sparse})),
                    {
                      {
                        // Sparse index
                        {0,1},
                        {0}
                      }
                    },
                    {1}
                    ),
           TestData(d5a("a", Format({Dense})),
                    {
                      {
                        // Dense index
                      }
                    },
                    {0, 1, 0, 0, 2}
                    ),
           TestData(d5a("a", Format({Sparse})),
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
    Values(TestData(d33a("A", Format({Dense,Dense})),
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
           TestData(d33a("A", Format({Sparse,Dense})),  // Blocked svec
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
           TestData(d33a("A", Format({Dense,Sparse})),  // CSR
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
           TestData(d33a("A", Format({Sparse,Sparse})),  // DCSR
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

INSTANTIATE_TEST_CASE_P(matrix_col, storage,
    Values(TestData(d33a("A", Format({Dense,Dense}, {1,0})),
                    {
                      {
                        // Dense index
                      },
                      {
                        // Dense index
                      }
                    },
                    {0, 0, 2,
                     1, 0, 0,
                     0, 0, 3}
                    ),
           TestData(d33a("A", Format({Sparse,Dense}, {1,0})),  // Blocked svec
                    {
                      {
                        // Sparse index
                        {0, 3},
                        {0, 1, 2},
                      },
                      {
                        // Dense index
                      }
                    },
                    {0, 0, 2,
                     1, 0, 0,
                     0, 0, 3}
                    ),
           TestData(d33a("A", Format({Dense,Sparse}, {1,0})),  // CSC
                    {
                      {
                        // Dense index
                      },
                      {
                        // Sparse index
                        {0, 1, 2, 3},
                        {2, 0, 2},
                      }
                    },
                    {2, 1, 3}
                    ),
           TestData(d33a("A", Format({Sparse,Sparse}, {1,0})),  // DCSC
                    {
                      {
                        // Sparse index
                        {0, 3},
                        {0, 1, 2},
                      },
                      {
                        // Sparse index
                        {0, 1, 2, 3},
                        {2, 0, 2},
                      }
                    },
                    {2, 1, 3}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(tensor3, storage,
    Values(TestData(d233a("A", Format({Dense,Dense,Dense})),
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
           TestData(d233a("A", Format({Sparse,Dense,Dense})),
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
           TestData(d233a("A", Format({Dense,Sparse,Dense})),
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
           TestData(d233a("A", Format({Sparse,Sparse,Dense})),
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
           TestData(d233a("A", Format({Dense,Dense,Sparse})),
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
           TestData(d233a("A", Format({Sparse,Dense,Sparse})),
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
           TestData(d233a("A", Format({Dense,Sparse,Sparse})),
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
           TestData(d233a("A", Format({Sparse,Sparse,Sparse})),
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

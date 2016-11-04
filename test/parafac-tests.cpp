#include "test.h"
#include "test_tensors.h"

#include "expr_factory.h"

#include "format.h"
#include "tensor.h"
#include "var.h"
#include "expr.h"
#include "expr_nodes.h"
#include "storage/storage.h"
#include "operator.h"

using namespace taco;

typedef std::tuple<ExprFactory*, std::vector<Tensor<double>>, 
                   Format, TensorData<double>> TestData; 

struct parafac : public TestWithParam<TestData> {};

TEST_P(parafac, eval) {
  std::vector<Tensor<double>> inputs = std::get<1>(GetParam());
  Format                      format = std::get<2>(GetParam());
  
  Tensor<double> tensor = (*std::get<0>(GetParam()))(inputs, format);

  tensor.eval();

  EXPECT_TRUE(std::get<3>(GetParam()).compare(tensor));
}

template <class ...Ts>
std::vector<Tensor<double>> packageInputs(Ts... inputs) {
  return {inputs...};
}

MatrixElwiseMultiplyFactory matElwiseMulFactory;
TensorInnerProductFactory   tenInnerProdFactory;

const Format scalarFormat;
const Format denseMatrixFormat({Dense, Dense});
const Format csfTensorFormat({Sparse, Sparse, Sparse});

INSTANTIATE_TEST_CASE_P(matrix_elwise_mul, parafac,
  Values(
    TestData(
      &matElwiseMulFactory,
      packageInputs(d33b("B", denseMatrixFormat), d33c("C", denseMatrixFormat)),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{0,1}, 200}, {{2,1}, 900}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(tensor_inner_prod, parafac,
  Values(
    TestData(
      &tenInnerProdFactory,
      packageInputs(d233a("B", csfTensorFormat)),
      scalarFormat,
      TensorData<double>({}, {{{}, 139}})
    )
  )
);

#if 0
INSTANTIATE_TEST_CASE_P(vector_neg, parafac,
    Values(
           TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    -d5a("b",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                        {5}
                      }
                    },
                    {0.0, -2.0, 0.0, 0.0, -3.0}
                    ),
           TestData(Tensor<double>("a",{5},Format({Sparse})),
                    {i},
                    -d5a("b",Format({Sparse}))(i),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {1,4}
                      },
                    },
                    {-2, -3}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix_neg, parafac,
    Values(
           TestData(Tensor<double>("a",{3,3},Format({Dense,Dense})),
                    {i,j},
                    -d33a("b",Format({Dense,Dense}))(i,j),
                    {
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    { 0, -2,  0,
                      0,  0,  0,
                     -3,  0, -4}
                    ),
           TestData(Tensor<double>("a",{3,3},Format({Sparse,Sparse})),
                    {i,j},
                    -d33a("b",Format({Sparse,Sparse}))(i,j),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {0,2}
                      },
                      {
                        // Sparse index
                        {0,1,3},
                        {1,0,2}
                      }
                    },
                    {-2, -3, -4}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(vector_mul, parafac,
    Values(
           TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    d5a("b",Format({Dense}))(i) *
                    d5b("c",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                        {5}
                      }
                    },
                    {0.0, 40.0, 0.0, 0.0, 0.0}
                    ),
           TestData(Tensor<double>("a",{5},Format({Sparse})),
                    {i},
                    d5a("b",Format({Sparse}))(i) *
                    d5b("c",Format({Sparse}))(i),
                    {
                      {
                        // Sparse index
                        {0,1},
                        {1}
                      }
                    },
                    {40.0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix_mul, parafac,
    Values(TestData(Tensor<double>("A",{3,3},Format({Sparse,Sparse})),
                    {i,j},
                    d33a("B",Format({Sparse,Sparse}))(i,j) *
                    d33b("C",Format({Sparse,Sparse}))(i,j),
                    {
                      {
                        // Sparse index
                        {0,1},
                        {0}
                      },
                      {
                        // Sparse index
                        {0,1},
                        {1}
                      }
                    },
                    {40.0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(tensor_mul, parafac,
    Values(
           TestData(Tensor<double>("A",{3,3,3},Format({Sparse,Sparse,Sparse})),
                    {i,j,k},
                    d233a("B",Format({Sparse,Sparse,Sparse}))(i,j,k) *
                    d233b("C",Format({Sparse,Sparse,Sparse}))(i,j,k),
                    {
                      {
                        // Sparse index
                        {0,1},
                        {1}
                      },
                      {
                        // Sparse index
                        {0,1},
                        {2}
                      },
                      {
                        // Sparse index
                        {0,1},
                        {0}
                      }
                    },
                    {300.0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(vector_add, parafac,
    Values(
           TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    d5a("b",Format({Dense}))(i) +
                    d5b("c",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                        {5}
                      }
                    },
                    {10.0, 22.0, 0.0, 0.0, 3.0}
                    ),
           TestData(Tensor<double>("a",{5},Format({Sparse})),
                    {i},
                    d5a("b",Format({Sparse}))(i) +
                    d5b("c",Format({Sparse}))(i),
                    {
                      {
                        // Sparse index
                        {0,3},
                        {0, 1, 4}
                      }
                    },
                    {10.0, 22.0, 3.0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix_add, parafac,
  Values(
         TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                  {i,j},
                  d33a("b",Format({Dense,Dense}))(i,j) +
                  d33b("c",Format({Dense,Dense}))(i,j),
                  {
                    {
                      // Dense index
                      {3}
                    },
                    {
                      // Dense index
                      {3}
                    }
                  },
                  {10, 22,  0,
                    0,  0,  0,
                    3, 30,  4}
                  ),
         TestData(Tensor<double>("A",{3,3},Format({Sparse,Sparse})),
                  {i,j},
                  d33a("b",Format({Sparse,Sparse}))(i,j) +
                  d33b("c",Format({Sparse,Sparse}))(i,j),
                  {
                    {
                      // Sparse index
                      {0,2},
                      {0,2}
                    },
                    {
                      // Sparse index
                      {0,2,5},
                      {0,1,0,1,2}
                    }
                  },
                  {10.0, 22.0, 3.0, 30.0, 4.0}
                  )
         )
);

INSTANTIATE_TEST_CASE_P(composite, parafac,
    Values(
           TestData(Tensor<double>("a",{5},Format({Sparse})),
                    {i},
                    d5a("b",Format({Sparse}))(i) +
                    (d5b("c",Format({Sparse}))(i) *
                     d5c("d",Format({Sparse}))(i)),
                    {
                      {
                        {0,2},
                        {1,4}
                      }
                    },
                    {2002.0, 3.0}
                    )
           )
);
#endif

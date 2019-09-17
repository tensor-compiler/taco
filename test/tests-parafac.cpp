#include "test.h"
#include "test_tensors.h"

#include "expr_factory.h"

#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/storage/storage.h"
#include "taco/format.h"

#include <cmath>

using namespace taco;

typedef std::tuple<ExprFactory*, std::vector<Tensor<double>>, 
                   Format, TensorData<double>> TestData; 

struct parafac : public TestWithParam<TestData> {};

TEST_P(parafac, eval) {
  std::vector<Tensor<double>> inputs = std::get<1>(GetParam());
  for (auto& tensor : inputs) {
    tensor.pack();
  }

  Format format = std::get<2>(GetParam());
  
  Tensor<double> tensor = (*std::get<0>(GetParam()))(inputs, format);
  tensor.evaluate();

  EXPECT_TRUE(std::get<3>(GetParam()).compare(tensor));
}

template <class ...Ts>
std::vector<Tensor<double>> packageInputs(Ts... inputs) {
  return {inputs...};
}

VectorElwiseSqrtFactory             vecElwiseSqrt;
MatrixElwiseMultiplyFactory         matElwiseMul;
MatrixMultiplyFactory               matMul;
MatrixTransposeMultiplyFactory      matTransposeMul;
MatrixColumnSquaredNormFactory      matColSquaredNorm;
MatrixColumnNormalizeFactory        matColNormalize;
MTTKRP1Factory                      MTTKRP1;
MTTKRP2Factory                      MTTKRP2;
MTTKRP3Factory                      MTTKRP3;
TensorSquaredNormFactory            tenSquaredNorm;
FactorizedTensorSquaredNormFactory  factTenSquaredNorm;
FactorizedTensorInnerProductFactory factTenInnerProd;
KroneckerFactory                    Kronecker;

const Format scalarFormat;
const Format denseVectorFormat({Dense});
const Format denseMatrixFormat({Dense, Dense});
const Format denseMatrixFormatTranspose({Dense, Dense}, {1,0});
const Format csfTensorFormat({Sparse, Sparse, Sparse});
const Format csfModeJTensorFormat({Sparse, Sparse, Sparse}, {1, 0, 2});
const Format csfModeKTensorFormat({Sparse, Sparse, Sparse}, {2, 0, 1});
const Format kronDenseFormat({Dense, Dense, Dense, Dense});
const Format kronCSRFormat({Dense, Sparse, Dense, Sparse});

INSTANTIATE_TEST_CASE_P(vector_elwise_sqrt, parafac,
  Values(
    TestData(
      &vecElwiseSqrt,
      packageInputs(d5c("v", denseVectorFormat)),
      denseVectorFormat,
      TensorData<double>({5}, {
        {{1}, std::sqrt(100)}, 
        {{3}, std::sqrt(200)},
        {{4}, std::sqrt(300)}
      })
    )
  )
);

INSTANTIATE_TEST_CASE_P(matrix_mul, parafac,
  Values(
    TestData(
      &matMul,
      packageInputs(d33a("B", denseMatrixFormat),
                    d33b("C", denseMatrixFormatTranspose)),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{2,0}, 30}, {{2,1}, 180}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(matrix_transpose_mul, parafac,
  Values(
    TestData(
      &matTransposeMul,
      packageInputs(d33a("B", denseMatrixFormatTranspose)),
      denseMatrixFormatTranspose,
      TensorData<double>({3,3}, {
        {{0,0}, 9}, 
        {{0,2}, 12},
        {{1,1}, 4},
        {{2,0}, 12},
        {{2,2}, 16}
      })
    )
  )
);

INSTANTIATE_TEST_CASE_P(matrix_column_squared_norm, parafac,
  Values(
    TestData(
      &matColSquaredNorm,
      packageInputs(d33a("B", denseMatrixFormatTranspose)),
      denseVectorFormat,
      TensorData<double>({3}, {{{0}, 9}, {{1}, 4}, {{2}, 16}})
    ),
    TestData(
      &matColSquaredNorm,
      packageInputs(d33b("B", denseMatrixFormatTranspose)),
      denseVectorFormat,
      TensorData<double>({3}, {{{0}, 100}, {{1}, 1300}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(matrix_column_normalize, parafac,
  Values(
    TestData(
      &matColNormalize,
      packageInputs(d33a("B", denseMatrixFormat), d3a("c", denseVectorFormat)),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{0,1}, 1.0}, {{2,0}, 1.0}, {{2,2}, 4.0}})
    )
//     ,
//    TestData(
//      &matColNormalize,
//      packageInputs(d33b("B", denseMatrixFormat), d3a("c", denseVectorFormat)),
//      denseMatrixFormat,
//      TensorData<double>({3,3}, {{{0,0}, 10.0/3.0}, {{0,1}, 10}, {{2,1}, 15}})
//    )
  )
);

INSTANTIATE_TEST_CASE_P(mttkrp1, parafac,
  Values(
    TestData(
      &MTTKRP1,
      packageInputs(
        d233a("B", csfTensorFormat),
        d33a("C", denseMatrixFormatTranspose),
        d33b("D", denseMatrixFormatTranspose)
      ),
      denseMatrixFormat,
      TensorData<double>({2,3}, {{{0,1}, 80}, {{1,0}, 180}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(mttkrp2, parafac,
  Values(
    TestData(
      &MTTKRP2,
      packageInputs(
        d333a("B", csfModeJTensorFormat),
        d33a("C", denseMatrixFormatTranspose),
        d33b("D", denseMatrixFormatTranspose)
      ),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{0,1}, 80}, {{2,1}, 240}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(mttkrp3, parafac,
  Values(
    TestData(
      &MTTKRP3,
      packageInputs(
        d333a("B", csfModeKTensorFormat),
        d33a("C", denseMatrixFormatTranspose),
        d33b("D", denseMatrixFormatTranspose)
      ),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{0,1}, 80}, {{1,1}, 120}, {{2,1}, 240}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(tensor_squared_norm, parafac,
  Values(
    TestData(
      &tenSquaredNorm,
      packageInputs(d233a("B", csfTensorFormat)),
      scalarFormat,
      TensorData<double>({}, {{{}, 139}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(factorized_tensor_squared_norm, parafac,
  Values(
    TestData(
      &factTenSquaredNorm,
      packageInputs(
        d3a("v", denseVectorFormat), 
        d33a("B", denseMatrixFormat),
        d33a("C", denseMatrixFormat),
        d33c("D", denseMatrixFormat)
      ),
      scalarFormat,
      TensorData<double>({}, {{{}, 780}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(factorized_tensor_inner_product, parafac,
  Values(
    TestData(
      &factTenInnerProd,
      packageInputs(
        d333a("X", csfTensorFormat),
        d3a("v", denseVectorFormat), 
        d33a("B", denseMatrixFormat),
        d33a("C", denseMatrixFormat),
        d33c("D", denseMatrixFormat)
      ),
      scalarFormat,
      TensorData<double>({}, {{{}, 160}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(kroneckerDense, parafac,
  Values(
    TestData(
      &Kronecker,
      packageInputs(
        d33a("B", taco::CSR),
        d33a("C", taco::CSR)
      ),
      kronDenseFormat,
      TensorData<double>({3,3,3,3},
                         {{{2,0,0,1}, 6}, {{2,0,2,0}, 9}, {{2,0,2,2}, 12},
                          {{0,1,0,1}, 4}, {{0,1,2,0}, 6}, {{0,1,2,2}, 8},
                          {{2,2,0,1}, 8}, {{2,2,2,0}, 12}, {{2,2,2,2}, 16}
      })
    )
  )
);

/*
INSTANTIATE_TEST_CASE_P(DISABLED_kroneckerCSR, parafac,
  Values(
    TestData(
      &Kronecker,
      packageInputs(
        d33a("B", taco::CSR),
        d33a("C", taco::CSR)
      ),
      kronCSRFormat,
      TensorData<double>({3,3,3,3},
                         {{{2,0,0,1}, 6}, {{2,0,2,0}, 9}, {{2,0,2,2}, 12},
                          {{0,1,0,1}, 4}, {{0,1,2,0}, 6}, {{0,1,2,2}, 8},
                          {{2,2,0,1}, 8}, {{2,2,2,0}, 12}, {{2,2,2,2}, 16}
      })
    )
  )
);
*/


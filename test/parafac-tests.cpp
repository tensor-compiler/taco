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

#include <cmath>

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

const Format scalarFormat;
const Format denseVectorFormat({Dense});
const Format denseMatrixFormat({Dense, Dense});
const Format csfTensorFormat({Sparse, Sparse, Sparse});

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

INSTANTIATE_TEST_CASE_P(DISABLED_matrix_mul, parafac,
  Values(
    TestData(
      &matMul,
      packageInputs(d33a("B", denseMatrixFormat), d33b("C", denseMatrixFormat)),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{2,0}, 30}, {{2,1}, 180}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(DISABLED_matrix_transpose_mul, parafac,
  Values(
    TestData(
      &matTransposeMul,
      packageInputs(d33a("B", denseMatrixFormat)),
      denseMatrixFormat,
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

INSTANTIATE_TEST_CASE_P(DISABLED_matrix_column_squared_norm, parafac,
  Values(
    TestData(
      &matColSquaredNorm,
      packageInputs(d33a("B", denseMatrixFormat)),
      denseVectorFormat,
      TensorData<double>({3}, {{{0}, 9}, {{1}, 4}, {{2}, 16}})
    ),
    TestData(
      &matColSquaredNorm,
      packageInputs(d33b("B", denseMatrixFormat)),
      denseVectorFormat,
      TensorData<double>({3}, {{{0}, 100}, {{1}, 1300}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(DISABLED_matrix_column_normalize, parafac,
  Values(
    TestData(
      &matColNormalize,
      packageInputs(d33a("B", denseMatrixFormat), d3a("c", denseVectorFormat)),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{0,1}, 1.0}, {{2,0}, 1.0}, {{2,2}, 4.0}})
    ),
    TestData(
      &matColNormalize,
      packageInputs(d33b("B", denseMatrixFormat), d3a("c", denseVectorFormat)),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{0,0}, 10.0/3.0}, {{0,1}, 10}, {{2,1}, 15}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(DISABLED_mttkrp1, parafac,
  Values(
    TestData(
      &MTTKRP1,
      packageInputs(
        d233a("B", csfTensorFormat),
        d33a("C", denseMatrixFormat),
        d33b("D", denseMatrixFormat)
      ),
      denseMatrixFormat,
      TensorData<double>({2,3}, {{{0,1}, 80}, {{1,0}, 180}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(DISABLED_mttkrp2, parafac,
  Values(
    TestData(
      &MTTKRP2,
      packageInputs(
        d333a("B", csfTensorFormat),
        d33a("C", denseMatrixFormat),
        d33b("D", denseMatrixFormat)
      ),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{0,1}, 80}, {{2,1}, 240}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(DISABLED_mttkrp3, parafac,
  Values(
    TestData(
      &MTTKRP3,
      packageInputs(
        d333a("B", csfTensorFormat),
        d33a("C", denseMatrixFormat),
        d33b("D", denseMatrixFormat)
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

INSTANTIATE_TEST_CASE_P(DISABLED_factorized_tensor_squared_norm, parafac,
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
      TensorData<double>({}, {{{}, 355600}})
    )
  )
);

INSTANTIATE_TEST_CASE_P(DISABLED_factorized_tensor_inner_product, parafac,
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


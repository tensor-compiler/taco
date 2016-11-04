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

  tensor.compile();
  tensor.assemble();
  tensor.compute();

  EXPECT_TRUE(std::get<3>(GetParam()).compare(tensor));
}

template <class ...Ts>
std::vector<Tensor<double>> packageInputs(Ts... inputs) {
  return {inputs...};
}

MatrixElwiseMultiplyFactory         matElwiseMul;
MatrixTransposeMultiplyFactory      matTransposeMul;
MTTKRPFactory                       MTTKRP;
TensorSquaredNormFactory            tenSquaredNorm;
FactorizedTensorSquaredNormFactory  factTenSquaredNorm;
FactorizedTensorInnerProductFactory factTenInnerProd;

const Format scalarFormat;
const Format denseVectorFormat({Dense});
const Format denseMatrixFormat({Dense, Dense});
const Format csfTensorFormat({Sparse, Sparse, Sparse});

INSTANTIATE_TEST_CASE_P(matrix_elwise_mul, parafac,
  Values(
    TestData(
      &matElwiseMul,
      packageInputs(d33b("B", denseMatrixFormat), d33c("C", denseMatrixFormat)),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{0,1}, 200}, {{2,1}, 900}})
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

INSTANTIATE_TEST_CASE_P(DISABLED_mttkrp, parafac,
  Values(
    TestData(
      &MTTKRP,
      packageInputs(
        d233a("B", csfTensorFormat),
        d33a("C", denseMatrixFormat),
        d33b("D", denseMatrixFormat)
      ),
      denseMatrixFormat,
      TensorData<double>({3,3}, {{{0,1}, 80}, {{1,0}, 180}})
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


#include "test.h"
#include "test_tensors.h"

#include "tensor.h"
#include "var.h"
#include "expr.h"
#include "packed_tensor.h"
#include "operator.h"

using namespace taco;

struct TestData {
  TestData(Tensor<double> tensor, const vector<Var> indexVars, Expr expr,
          PackedTensor::Indices expectedIndices, vector<double> expectedValues)
      : tensor(tensor),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
    tensor(indexVars) = expr;
  }

  Tensor<double>        tensor;
  PackedTensor::Indices expectedIndices;
  vector<double>        expectedValues;
};


struct expr : public TestWithParam<TestData> {};

TEST_P(expr, eval) {
  Tensor<double> tensor = GetParam().tensor;

  std::cout << tensor.getName() << "(" << util::join(tensor.getIndexVars()) << ")"
            << " = " << tensor.getExpr() << std::endl;

  tensor.compile();
  tensor.assemble();
  tensor.evaluate();

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
  ASSERT_VECTOR_EQ(expectedValues, values);
}

Var i("i"), j("j"), k("k"), l("l");

INSTANTIATE_TEST_CASE_P(neg, expr,
                        Values(TestData(Tensor<double>("a", {5}, "d"),
                                        {i},
                                        -d5a("b", "d")(i),
                                        {
                                          {
                                            // Dense index
                                          }
                                        },
                                        {0.0, -1.0, 0.0, 0.0, -2.0}
                                        ),
                               TestData(Tensor<double>("a", {5}, "d"),
                                        {i,j},
                                        -d33a("b", "dd")(i,j),
                                        {
                                          {
                                            // Dense index
                                          },
                                          {
                                            // Dense index
                                          }
                                        },
                                        { 0, -1,  0,
                                          0,  0,  0,
                                         -2,  0, -3}
                                        )
                               )
                        );

INSTANTIATE_TEST_CASE_P(add, expr,
                        Values(TestData(Tensor<double>("a", {5}, "d"),
                                        {i},
                                        d5a("b","d")(i) +
                                        d5b("c","d")(i),
                                        {
                                          {
                                            // Dense index
                                          }
                                        },
                                        {0.0, -1.0, 0.0, 0.0, -2.0}
                                        ),
                               TestData(Tensor<double>("a", {5}, "d"),
                                        {i,j},
                                        d33a("b","dd")(i,j) +
                                        d33b("c","dd")(i,j),
                                        {
                                          {
                                            // Dense index
                                          },
                                          {
                                            // Dense index
                                          }
                                        },
                                        { 0, -1,  0,
                                          0,  0,  0,
                                         -2,  0, -3}
                                        )
                               )
                        );

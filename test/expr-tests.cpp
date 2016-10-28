#include "test.h"
#include "test_tensors.h"
#include "expr-tests.h"

#include "tensor.h"
#include "var.h"
#include "expr.h"
#include "expr_nodes.h"
#include "packed_tensor.h"
#include "operator.h"

using namespace taco;

typedef int                     IndexType;
typedef std::vector<IndexType>  IndexArray; // Index values
typedef std::vector<IndexArray> Index;      // [0,2] index arrays per Index
typedef std::vector<Index>      Indices;    // One Index per level

struct TestData {
  TestData(Tensor<double> tensor, const vector<Var> indexVars, Expr expr,
          Indices expectedIndices, vector<double> expectedValues)
      : tensor(tensor),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
    tensor(indexVars) = expr;
  }

  Tensor<double> tensor;
  Indices        expectedIndices;
  vector<double> expectedValues;
};

static ostream &operator<<(ostream& os, const TestData& data) {
  os << data.tensor.getName() << ": "
     << util::join(data.tensor.getDimensions(), "x")
     << " (" << data.tensor.getFormat() << ")";
  return os;
}

struct expr : public TestWithParam<TestData> {};

TEST_P(expr, eval) {
  Tensor<double> tensor = GetParam().tensor;

  for (auto& operand : internal::getOperands(tensor.getExpr())) {
    std::cout << operand << std::endl;
  }

  tensor.printIterationSpace();

//  tensor.compile();
//  tensor.assemble();
//  tensor.evaluate();
//
//  auto tensorPack = tensor.getPackedTensor();
//  ASSERT_NE(nullptr, tensorPack);
//
//  // Check that the indices are as expected
//  auto& expectedIndices = GetParam().expectedIndices;
//  auto&         indices = tensorPack->getIndices();
//  ASSERT_EQ(expectedIndices.size(), indices.size());
//
//  for (size_t i=0; i < indices.size(); ++i) {
//    auto expectedIndex = expectedIndices[i];
//    auto         index = indices[i];
//    ASSERT_EQ(expectedIndex.size(), index.size());
//    for (size_t j=0; j < index.size(); ++j) {
//      ASSERT_VECTOR_EQ(expectedIndex[j], index[j]);
//    }
//  }
//
//  auto& expectedValues = GetParam().expectedValues;
//  ASSERT_EQ(expectedValues.size(), tensorPack->getNnz());
//  auto values = tensorPack->getValues();
//  ASSERT_VECTOR_EQ(expectedValues, values);
}

Var i("i"), j("j"), k("k"), l("l");

INSTANTIATE_TEST_CASE_P(vector_neg, expr,
    Values(TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    -d5a("b",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                      }
                    },
                    {0.0, -1.0, 0.0, 0.0, -2.0}
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
                    {-1, -2}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix_neg, expr,
    Values(TestData(Tensor<double>("a",{3,3},Format({Dense,Dense})),
                    {i,j},
                    -d33a("b",Format({Dense,Dense}))(i,j),
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
                    ),
           TestData(Tensor<double>("a",{3,3},Format({Dense,Sparse})),
                    {i,j},
                    -d33a("b",Format({Dense,Sparse}))(i,j),
                    {
                      {
                        // Dense index
                      },
                      {
                        {0,1,1,3},
                        {1,0,2}
                      }
                    },
                    {-1, -2, -3}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(vector_mul, expr,
    Values(TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    d5a("b",Format({Dense}))(i) *
                    d5b("c",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                      }
                    },
                    {0.0, 20.0, 0.0, 0.0, 0.0}
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
                    {20.0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix_mul, expr,
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
                    {20.0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(vector_add, expr,
    Values(
           TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    d5a("b",Format({Dense}))(i) +
                    d5b("c",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                      }
                    },
                    {10.0, 21.0, 0.0, 0.0, 2.0}
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
                    {10.0, 21.0, 2.0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix_add, expr,
  Values(TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                  {i,j},
                  d33a("b",Format({Dense,Dense}))(i,j) +
                  d33b("c",Format({Dense,Dense}))(i,j),
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

INSTANTIATE_TEST_CASE_P(composite, expr,
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
                    {2001.0, 2.0}
                    )
           )
);

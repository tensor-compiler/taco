#include "test.h"
#include "test_tensors.h"

#include "tensor.h"
#include "var.h"
#include "expr.h"
#include "expr_nodes.h"
#include "storage/storage.h"
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

//  tensor.printIterationSpace();

  tensor.compile();
  tensor.assemble();
  tensor.compute();

//  tensor.printIR(cout);

  auto storage = tensor.getStorage();
  ASSERT_TRUE(storage.defined());
  auto levels = storage.getFormat().getLevels();

  // Check that the indices are as expected
  auto& expectedIndices = GetParam().expectedIndices;
  auto size = storage.getSize();

  for (size_t i=0; i < levels.size(); ++i) {
    auto expectedIndex = expectedIndices[i];
    auto levelIndex = storage.getLevelIndex(i);
    auto levelIndexSize = size.levelIndices[i];

    switch (levels[i].getType()) {
      case LevelType::Dense: {
        iassert(expectedIndex.size() == 1);
        ASSERT_ARRAY_EQ(expectedIndex[0], {levelIndex.ptr, levelIndexSize.ptr});
        ASSERT_EQ(nullptr, levelIndex.idx);
        ASSERT_EQ(0u, levelIndexSize.idx);
        break;
      }
      case LevelType::Sparse: {
        iassert(expectedIndex.size() == 2);
        ASSERT_ARRAY_EQ(expectedIndex[0], {levelIndex.ptr, levelIndexSize.ptr});
        ASSERT_ARRAY_EQ(expectedIndex[1], {levelIndex.idx, levelIndexSize.idx});
        break;
      }
      case LevelType::Fixed:
        break;
    }
  }

  auto& expectedValues = GetParam().expectedValues;
  ASSERT_EQ(expectedValues.size(), storage.getSize().values);
  ASSERT_ARRAY_EQ(expectedValues, {storage.getValues(), size.values});
}

Var i("i"), j("j"), k("k"), l("l");

INSTANTIATE_TEST_CASE_P(vector_neg, expr,
    Values(
//           TestData(Tensor<double>("a",{5},Format({Dense})),
//                    {i},
//                    -d5a("b",Format({Dense}))(i),
//                    {
//                      {
//                        // Dense index
//                      }
//                    },
//                    {0.0, -1.0, 0.0, 0.0, -2.0}
//                    ),
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

INSTANTIATE_TEST_CASE_P(matrix_neg, expr,
    Values(
//           TestData(Tensor<double>("a",{3,3},Format({Dense,Dense})),
//                    {i,j},
//                    -d33a("b",Format({Dense,Dense}))(i,j),
//                    {
//                      {
//                        // Dense index
//                      },
//                      {
//                        // Dense index
//                      }
//                    },
//                    { 0, -1,  0,
//                      0,  0,  0,
//                     -2,  0, -3}
//                    ),
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

INSTANTIATE_TEST_CASE_P(vector_mul, expr,
    Values(
//           TestData(Tensor<double>("a",{5},Format({Dense})),
//                    {i},
//                    d5a("b",Format({Dense}))(i) *
//                    d5b("c",Format({Dense}))(i),
//                    {
//                      {
//                        // Dense index
//                      }
//                    },
//                    {0.0, 20.0, 0.0, 0.0, 0.0}
//                    ),
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
                    {40.0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(tensor_mul, expr,
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

INSTANTIATE_TEST_CASE_P(vector_add, expr,
    Values(
//           TestData(Tensor<double>("a",{5},Format({Dense})),
//                    {i},
//                    d5a("b",Format({Dense}))(i) +
//                    d5b("c",Format({Dense}))(i),
//                    {
//                      {
//                        // Dense index
//                      }
//                    },
//                    {10.0, 21.0, 0.0, 0.0, 2.0}
//                    ),
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

INSTANTIATE_TEST_CASE_P(matrix_add, expr,
  Values(
//         TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
//                  {i,j},
//                  d33a("b",Format({Dense,Dense}))(i,j) +
//                  d33b("c",Format({Dense,Dense}))(i,j),
//                  {
//                    {
//                      // Dense index
//                      {3}
//                    },
//                    {
//                      // Dense index
//                      {3}
//                    }
//                  },
//                  {10, 21,  0,
//                    0,  0,  0,
//                    2, 30, 33}
//                  )
//         ),
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
                    {2002.0, 3.0}
                    )
           )
);

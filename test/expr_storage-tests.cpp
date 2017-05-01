#include "test.h"
#include "test_tensors.h"

#include "taco/tensor.h"
#include "taco/expr.h"
#include "taco/operator.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/storage/storage.h"

using namespace taco;

namespace expr_storage_tests {

typedef int                     IndexType;
typedef std::vector<IndexType>  IndexArray; // Index values
typedef std::vector<IndexArray> Index;      // [0,2] index arrays per Index
typedef std::vector<Index>      Indices;    // One Index per level

Var i("i"), j("j"), m("m"), n("n");
Var k("k", Var::Sum), l("l", Var::Sum);


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

TEST_P(expr, storage) {
  Tensor<double> tensor = GetParam().tensor;
  packOperands(tensor);

  tensor.compile();
  tensor.assemble();
  tensor.compute();

  SCOPED_TRACE(tensor.getName() + "(" + util::join(tensor.getIndexVars(),",") +
               ") = " + toString(tensor.getExpr()));

  auto storage = tensor.getStorage();
  auto levels = storage.getFormat().getLevels();

  // Check that the indices are as expected
  auto& expectedIndices = GetParam().expectedIndices;
  taco_iassert(expectedIndices.size() == levels.size());
  auto size = storage.getSize();

  for (size_t i=0; i < levels.size(); ++i) {
    auto expectedIndex = expectedIndices[i];
    auto index = storage.getDimensionIndex(i);

    switch (levels[i].getType()) {
      case LevelType::Dense: {
        taco_iassert(expectedIndex.size() == 1) <<
            "Dense indices have a ptr array";
        ASSERT_EQ(1u, index.size());
        ASSERT_ARRAY_EQ(expectedIndex[0], {index[0], size.numIndexValues(i,0)});
        break;
      }
      case LevelType::Sparse:
      case LevelType::Fixed: {
        taco_iassert(expectedIndex.size() == 2);
        ASSERT_EQ(2u, index.size());
        ASSERT_ARRAY_EQ(expectedIndex[0], {index[0], size.numIndexValues(i,0)});
        ASSERT_ARRAY_EQ(expectedIndex[1], {index[1], size.numIndexValues(i,1)});
        break;
      }
    }
  }

  auto& expectedValues = GetParam().expectedValues;
  ASSERT_EQ(expectedValues.size(), storage.getSize().numValues());
  ASSERT_ARRAY_EQ(expectedValues, {storage.getValues(), size.numValues()});
}

INSTANTIATE_TEST_CASE_P(scalar, expr,
    Values(
           TestData(Tensor<double>("a",{},Format()),
                    {},
                    -da("b",Format())(),
                    {
                    },
                    {-2}
                    ),
           TestData(Tensor<double>("a",{},Format()),
                    {},
                    da("b",Format())() *
                    db("c",Format())(),
                    {
                    },
                    {20}
                    ),
           TestData(Tensor<double>("a",{},Format()),
                    {},
                    da("b",Format())() +
                    db("c",Format())(),
                    {
                    },
                    {12}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(vector_neg, expr,
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

INSTANTIATE_TEST_CASE_P(vector_scalar, expr,
    Values(
           TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    d5a("b",Format({Dense}))(i) * da("c",Format())(),
                    {
                      {
                        // Dense index
                        {5}
                      }
                    },
                    {0.0, 4.0, 0.0, 0.0, 6.0}
                    ),
           TestData(Tensor<double>("a",{5},Format({Sparse})),
                    {i},
                    d5a("b",Format({Sparse}))(i) * da("c",Format())(),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {1,4}
                      }
                    },
                    {4.0, 6.0}
                    ),
           TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    d5a("b",Format({Dense}))(i) + da("c",Format())(),
                    {
                      {
                        // Dense index
                        {5}
                      }
                    },
                    {2.0, 4.0, 2.0, 2.0, 5.0}
                    )
           )
);


INSTANTIATE_TEST_CASE_P(vector_elmul, expr,
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
           TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    d5a("b",Format({Sparse}))(i) *
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

INSTANTIATE_TEST_CASE_P(vector_add, expr,
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
           TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    d5a("b",Format({Sparse}))(i) +
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
                    d5b("c",Format({Dense}))(i),
                    {
                      {
                        // Sparse index
                        {0,5},
                        {0, 1, 2, 3, 4}
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

INSTANTIATE_TEST_CASE_P(vector_composites, expr,
    Values(
           TestData(Tensor<double>("a",{8},Format({Dense})),
                    {i},
                    (d8a("b",Format({Sparse}))(i) +
                     d8b("c",Format({Dense}))(i)) *
                    d8c("d",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                        {8}
                      }
                    },
                    {0, 200, 0, 6000, 0, 1200, 0, 0}
                    ),
           TestData(Tensor<double>("a",{8},Format({Dense})),
                    {i},
                    (d8a("b",Format({Sparse}))(i) +
                     d8b("c",Format({Sparse}))(i)) *
                    d8c("d",Format({Sparse}))(i),
                    {
                      {
                        // Dense index
                        {8}
                      }
                    },
                    {0, 200, 0, 6000, 0, 1200, 0, 0}
                    ),
           TestData(Tensor<double>("a",{8},Format({Dense})),
                    {i},
                    d8a("b",Format({Sparse}))(i) *
                    (d8b("c",Format({Dense}))(i) +
                     d8c("d",Format({Dense}))(i)),
                    {
                      {
                        // Dense index
                        {8}
                      }
                    },
                    {10, 200, 60, 0, 0, 1200, 0, 0}
           //         ),
           //TestData(Tensor<double>("a",{8},Format({Dense})),
           //         {i}, 
           //         (d8b("c",Format({Sparse}))(i) +
           //          d8c("d",Format({Dense}))(i)) * 
           //          d8a("b",Format({Sparse}))(i),
           //         {
           //           {
           //             // Dense index
           //             {8}
           //           }
           //         },
           //         {10, 200, 60, 0, 0, 1200, 0, 0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix_neg, expr,
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

INSTANTIATE_TEST_CASE_P(matrix_elmul, expr,
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

INSTANTIATE_TEST_CASE_P(DISABLED_residual, expr,
  Values(
          TestData(Tensor<double>("r",{3},Format({Dense})),
                  {i},
				  d3a("b",Format({Dense}))(i) -
                  d33a("A",Format({Dense,Dense}))(i,k) *
                  d3b("x",Format({Dense}))(k),
                  {
                    {
                      // Dense index
                      {3}
                    },
                  },
                  {3, 2, -17}
                  ),
          TestData(Tensor<double>("r",{3},Format({Dense})),
                  {i},
				  d3a("b",Format({Dense}))(i) -
                  d33a("A",Format({Sparse,Sparse}))(i,k) *
                  d3b("x",Format({Dense}))(k),
                  {
                    {
                      // Dense index
                      {3}
                    },
                  },
                  {3, 2, -17}
                  )
		)
);

INSTANTIATE_TEST_CASE_P(DISABLED_interpol, expr,
  Values(
          TestData(Tensor<double>("B",{3,5},Format({Dense,Dense})),
                  {i,j},
				  d35a("I",Format({Dense,Dense}))(i,k) *
                  d55a("A",Format({Dense,Dense}))(k,j),
                  {
                    {
                      // Dense index
                      {3}
                    },
					{
					  {5}
					}
                  },
                  {18,0,0,0,28,
                    0,0,0,0,0,
                   26,0,0,0,0}
                  )
		)
);

INSTANTIATE_TEST_CASE_P(DISABLED_MG, expr,
  Values(
          TestData(Tensor<double>("B",{3,3},Format({Dense,Dense})),
                  {i,j},
				  0.25 *
				  d35a("I",Format({Dense,Dense}))(i,k) *
                  d55a("A",Format({Dense,Dense}))(k,l) *
				  d53a("It",Format({Dense,Dense}))(l,j),
                  {
                    {
                      // Dense index
                      {3}
                    },
					{
					  {3}
					}
                  },
                  {9,0,13.5,
                    0,0,0,
				   13,0,19.5}
                  )
		)
);

INSTANTIATE_TEST_CASE_P(matrix_add, expr,
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
         TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                  {i,j},
                  d33a("B",Format({Dense,Sparse}))(i,j) +
                  d33b("C",Format({Dense,Sparse}))(i,j),
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
         TestData(Tensor<double>("A",{3,3},Format({Dense,Sparse})),
                  {i,j},
                  d33a("B",Format({Dense,Sparse}))(i,j) +
                  d33b("C",Format({Dense,Sparse}))(i,j),
                  {
                    {
                      // Dense
                      {3}
                    },
                    {
                      // Sparse index
                      {0,2,2,5},
                      {0,1,0,1,2}
                    }
                  },
                  {10.0, 22.0, 3.0, 30.0, 4.0}
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

INSTANTIATE_TEST_CASE_P(tensor_elmul, expr,
    Values(
           TestData(Tensor<double>("A",{3,3,3},Format({Sparse,Sparse,Sparse})),
                    {i,j,m},
                    d233a("B",Format({Sparse,Sparse,Sparse}))(i,j,m) *
                    d233b("C",Format({Sparse,Sparse,Sparse}))(i,j,m),
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

INSTANTIATE_TEST_CASE_P(composite, expr,
    Values(
           TestData(Tensor<double>("a",{5},Format({Sparse})),
                    {i},
                    d5b("b",Format({Sparse}))(i) *
                    d5c("c",Format({Sparse}))(i) +
                    d5a("d",Format({Sparse}))(i),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {1,4}
                      }
                    },
                    {2002.0, 3.0}
                    ),
           TestData(Tensor<double>("a",{5},Format({Sparse})),
                    {i},
                    d5a("b",Format({Sparse}))(i) +
                    (d5b("c",Format({Sparse}))(i) *
                     d5c("d",Format({Sparse}))(i)),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {1,4}
                      }
                    },
                    {2002.0, 3.0}
                    ),
           TestData(Tensor<double>("a",{5},Format({Dense})),
                    {i},
                    d5a("b",Format({Sparse}))(i) +
                    (d5b("c",Format({Sparse}))(i) *
                     d5c("d",Format({Sparse}))(i)),
                    {
                      {
                        // Dense index
                        {5}
                      }
                    },
                    {0.0, 2002.0, 0.0, 0.0, 3.0}
                    )
           )
);

// a = alpha(B+C)d
INSTANTIATE_TEST_CASE_P(matrix_add_vec_mul_composite, expr,
    Values(
           TestData(Tensor<double>("a",{3},Format({Dense})),
                    {i},
                    da("alpha", Format())() *
                    (d33a("B", Format({Dense,Sparse}))(i,k) +
                     d33b("C", Format({Dense,Sparse}))(i,k)) *
                    d3a("d",Format({Dense}))(k)
                    ,
                    {
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {148, 0, 146}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(axpy_3x3, expr,
    Values(
           TestData(Tensor<double>("a",{3},Format({Dense})),
                    {i},
                    d33a("B",Format({Dense,Dense}))(i,k) *
                     d3a("c",Format({Dense}))(k) +
                     d3b("d",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {6, 0, 16}
                    ),
           TestData(Tensor<double>("a",{3},Format({Dense})),
                    {i},
                    d33a("B",Format({Dense,Sparse}))(i,k) *
                     d3a("c",Format({Dense}))(k) +
                     d3b("d",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {6, 0, 16}
                    )
//                    ,
//           TestData(Tensor<double>("a",{3},Format({Dense})),
//                    {i},
//                    d33a("B",Format({Sparse,Sparse}))(i,k) *
//                     d3a("c",Format({Sparse}))(k) +
//                     d3b("d",Format({Sparse}))(i),
//                    {
//                      {
//                        // Dense index
//                        {3}
//                      }
//                    },
//                    {6, 0, 16}
//                    )
           )
);

INSTANTIATE_TEST_CASE_P(DISABLED_axpy_4x4, expr,
    Values(
           TestData(Tensor<double>("a",{4},Format({Dense})),
                    {i},
                    d44a("B",Format({Dense,Dense}))(i,k) *
                     d4a("c",Format({Dense}))(k) +
                     d4b("d",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                        {4}
                      }
                    },
                    {60, 0, 100, 20}
                    ),
           TestData(Tensor<double>("a",{4},Format({Dense})),
                    {i},
                    d44a("B",Format({Dense,Sparse}))(i,k) *
                     d4a("c",Format({Dense}))(k) +
                     d4b("d",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                        {4}
                      }
                    },
                    {60, 0, 100, 20}
                    ),
           TestData(Tensor<double>("a",{4},Format({Dense})),
                    {i},
                    d44a("B",Format({Dense,Sparse}))(i,k) *
                     d4a("c",Format({Sparse}))(k) +
                     d4b("d",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                        {4}
                      }
                    },
                    {60, 0, 100, 20}
                    )
//                    ,
//           TestData(Tensor<double>("a",{4},Format({Dense})),
//                    {i},
//                    d44a("B",Format({Sparse,Sparse}))(i,k) *
//                     d4a("c",Format({Sparse}))(k) +
//                     d4b("d",Format({Dense}))(i),
//                    {
//                      {
//                        // Dense index
//                        {4}
//                      }
//                    },
//                    {60, 0, 100, 20}
//                    ),
//           TestData(Tensor<double>("a",{4},Format({Dense})),
//                    {i},
//                    d44a("B",Format({Sparse,Sparse}))(i,k) *
//                     d4a("c",Format({Sparse}))(k) +
//                     d4b("d",Format({Sparse}))(i),
//                    {
//                      {
//                        // Dense index
//                        {4}
//                      }
//                    },
//                    {60, 0, 100, 20}
//                    )
           )
);

INSTANTIATE_TEST_CASE_P(DISABLED_axpy_3x3_scaled, expr,
    Values(
           TestData(Tensor<double>("a",{3},Format({Dense})),
                    {i},
                      da("alpha",Format())() *
                    d33a("B",Format({Dense,Sparse}))(i,k) *
                     d3a("c",Format({Dense}))(k) +
                      db("beta",Format())() *
                     d3b("d",Format({Dense}))(i),
                    {
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {28, 0, 56}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(vector_inner, expr,
    Values(
           TestData(Tensor<double>("a",{},Format()),
                    {},
                    d5a("b",Format({Dense}))(k) *
                    d5b("c",Format({Dense}))(k),
                    {
                    },
                    {40.0}
                    ),
           TestData(Tensor<double>("a",{},Format()),
                    {},
                    d5a("b",Format({Sparse}))(k) *
                    d5b("c",Format({Sparse}))(k),
                    {
                    },
                    {40.0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(spmv, expr,
    Values(
           TestData(Tensor<double>("a",{3},Format({Dense})),
                    {i},
                    d33a("B",Format({Dense, Dense}))(i,k) *
                    d3b("c",Format({Dense}))(k),
                    {
                      {
                        // Dense index
                        {3}
                      },
                    },
                    {0,0,18}
                    ),
           TestData(Tensor<double>("a",{3},Format({Dense})),
                    {i},
                    d33a("B",Format({Dense, Sparse}))(i,k) *
                    d3b("c",Format({Dense}))(k),
                    {
                      {
                        // Dense index
                        {3}
                      },
                    },
                    {0,0,18}
                    ),
           TestData(Tensor<double>("a",{3},Format({Dense})),
                    {i},
                    d33a("B",Format({Dense, Sparse}))(i,k) *
                    d3b("c",Format({Sparse}))(k),
                    {
                      {
                        // Dense index
                        {3}
                      },
                    },
                    {0,0,18}
                    ),
           TestData(Tensor<double>("a",{3},Format({Sparse})),
                    {i},
                    d33a("B",Format({Sparse, Sparse}))(i,k) *
                    d3b("c",Format({Sparse}))(k),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {0,2}
                      },
                    },
                    {0,18}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(bspmv, expr,
    Values(
           TestData(Tensor<double>("a", {3,2}, Format({Dense,Dense})),
                    {i,j},
                    d3322a("B",Format({Dense,Sparse,Dense,Dense}))(i,k,j,l) *
                    d32b("c",Format({Dense,Dense}))(k,l),
                    {
                      {
                        // Dense index
                        {3},
                      },
                      {
                        // Dense index
                        {2},
                      }
                    },
                    {88.2, 96.4, 0.0, 0.0, 319.4, 335.8}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix_sum, expr,
    Values(
           TestData(Tensor<double>("a",{},Format()),
                    {},
                    d33a("B",Format({Dense, Dense}))(k,l),
                    {},
                    {9.0}
                    ),
           TestData(Tensor<double>("a",{},Format()),
                    {},
                    d33a("B",Format({Sparse, Sparse}))(k,l),
                    {},
                    {9.0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix_mul, expr,
    Values(
           TestData(Tensor<double>("a",{3,3},Format({Dense,Dense})),
                    {i,j},
                    d33a("B",Format({Dense, Dense}))(i,k) *
                    d33b("C",Format({Dense, Dense}, {1,0}))(k,j),
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
                    {  0,   0,   0,
                       0,   0,   0,
                      30, 180,   0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(tensor_vector_mul, expr,
    Values(
           TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                    {i,j},
                    d333a("B",Format({Dense, Dense, Dense}))(i,j,k) *
                    d3b("c",Format({Dense}))(k),
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
                    {4,  0, 12,
                     0,  0, 33,
                     0, 24,  0}
                    ),
           TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                    {i,j},
                    d333a("B",Format({Sparse, Sparse, Sparse}))(i,j,k) *
                    d3b("c",Format({Dense}))(k),
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
                    {4,  0, 12,
                     0,  0, 33,
                     0, 24,  0}
                    ),
           TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                    {i,j},
                    d333a("B",Format({Sparse, Sparse, Sparse}))(i,j,k) *
                    d3b("c",Format({Sparse}))(k),
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
                    {4,  0, 12,
                     0,  0, 33,
                     0, 24,  0}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(tensor_matrix_mul, expr,
    Values(
           TestData(Tensor<double>("A",{2,3},Format({Dense,Dense})),
                    {i,j},
                    d233a("B",Format({Sparse, Sparse, Sparse}))(i,j,k) *
                     d33a("C",Format({Dense, Dense}))(j,k),
                    {
                      {
                        // Dense index
                        {2}
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {6,  0, 16,
                     10, 0, 46}
                    ),
           TestData(Tensor<double>("A",{2,3,3},Format({Dense,Dense, Dense})),
                    {i,m,j},
                    d233a("B",Format({Sparse, Sparse, Sparse}))(i,m,l) *
                     d33a("C",Format({Dense, Dense}, {1,0}))(l,j),
                    {
                      {
                        // Dense index
                        {2}
                      },
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {  0,   4,   0,
                       0,   0,   0,
                      12,   0,  16,

                       0,   0,   0,
                       0,   0,   0,
                      21,  12,  28}
                    )
           )
);

// A(i,j) = B(i,k,l) * C(k,j) * B(l,j)
INSTANTIATE_TEST_CASE_P(mttkrp, expr,
    Values(
           TestData(Tensor<double>("A",{2,3},Format({Dense,Dense})),
                    {i,j},
                    d233a("B",Format({Sparse, Sparse, Sparse}))(i,k,l) *
                    d33a("C",Format({Dense, Dense}, {1,0}))(k,j) *
                    d33b("D",Format({Dense, Dense}, {1,0}))(l,j),
                    {
                      {
                        // Dense index
                        {2}
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {  0,  80,   0,
                     180,   0,   0}
                    )
           )
);

// A(i,j) =  b(i)         * D(i,j)
// A(i,j) = (b(i) + c(i)) * D(i,j)
// A(i,j) = b(i) * D(i,j) * c(i)
// A(i,j,k) = b(i) * D(i,j,k) * c(j)
INSTANTIATE_TEST_CASE_P(emit_avail_exprs, expr,
    Values(
           TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                    {i,j},
                    d3a("b",Format({Dense}))(i) *
                    d33a("D",Format({Dense, Dense}, {0,1}))(i,j),
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
                    {  0,   6,   0,
                       0,   0,   0,
                       3,   0,   4}
                    ),
           TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                    {i,j},
                    (d3a("b",Format({Dense}))(i) +
                     d3b("c",Format({Dense}))(i)) *
                    d33a("D",Format({Dense, Dense}, {0,1}))(i,j),
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
                    {  0,  10,   0,
                       0,   0,   0,
                      12,   0,  16}
                    ),
           TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                    {i,j},
                    d3a("b",Format({Dense}))(i) *
                    d33a("D",Format({Dense, Dense}, {0,1}))(i,j) *
                    d3b("c",Format({Dense}))(i),
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
                    {  0,  12,   0,
                       0,   0,   0,
                       9,   0,  12}
                    ),
           TestData(Tensor<double>("A",{3,3,3},Format({Dense,Dense,Dense})),
                    {i,j,m},
                    d3a("b",Format({Dense}))(i) *
                    d333a("D",Format({Dense, Dense,Dense}, {0,1,2}))(i,j,m) *
                    d3b("c",Format({Dense}))(j),
                    {
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    { 12,  18,   0,
                       0,   0,   0,
                       0,   0,  36,

                       0,  20,   0,
                       0,   0,   0,
                      36,   0,  42,

                       0,   0,   0,
                       0,   0,   0,
                       0,  27,   0}
                    )
           )
);

// A(i,j) = b(j)
// A(i,j) = b(i)
INSTANTIATE_TEST_CASE_P(DISABLED_copy, expr,
    Values(
           TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                    {i,j},
                    d3b("b",Format({Dense}))(j),
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
                    {  2,   0,   3,
                       2,   0,   3,
                       2,   0,   3}
                    ),
           TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                    {i,j},
                    d3b("b",Format({Sparse}))(j),
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
                    {  2,   0,   3,
                       2,   0,   3,
                       2,   0,   3}
                    ),
           TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                    {i,j},
                    d3b("b",Format({Dense}))(i),
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
                    {  2,   2,   2,
                       0,   0,   0,
                       3,   3,   3}
                    ),
           TestData(Tensor<double>("A",{3,3},Format({Dense,Dense})),
                    {i,j},
                    d3b("b",Format({Sparse}))(i),
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
                    {  2,   2,   2,
                       0,   0,   0,
                       3,   3,   3}
                    )
           )
);

}

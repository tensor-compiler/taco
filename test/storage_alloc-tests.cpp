#include "test.h"
#include "test_tensors.h"

#include "taco/tensor.h"
#include "taco/expr.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/storage/storage.h"

using namespace taco;

namespace storage_alloc_tests {

typedef int                     IndexType;
typedef std::vector<IndexType>  IndexArray; // Index values
typedef std::vector<IndexArray> Index;      // [0,2] index arrays per Index
typedef std::vector<Index>      Indices;    // One Index per level

struct TestData {
  TestData(Tensor<double> tensor, size_t allocSize, bool assembleWhileCompute,
           const vector<IndexVar>& indexVars, IndexExpr expr,
           Indices expectedIndices, vector<double> expectedValues)
      : tensor(tensor), assembleWhileCompute(assembleWhileCompute), 
      expectedIndices(expectedIndices), expectedValues(expectedValues) {
    tensor(indexVars) = expr;
    tensor.setAllocSize(allocSize);
  }

  Tensor<double> tensor;
  bool           assembleWhileCompute;
  Indices        expectedIndices;
  vector<double> expectedValues;
};

static ostream &operator<<(ostream& os, const TestData& data) {
  os << data.tensor.getName() << ": "
     << util::join(data.tensor.getDimensions(), "x")
     << " (" << data.tensor.getFormat() << ")";
  return os;
}

struct alloc : public TestWithParam<TestData> {};

TEST_P(alloc, storage) {
  Tensor<double> tensor = GetParam().tensor;
  packOperands(tensor);

  tensor.compile(GetParam().assembleWhileCompute);
  tensor.assemble();
  tensor.compute();

  auto& expectedIndices = GetParam().expectedIndices;
  auto& expectedValues = GetParam().expectedValues;
  ASSERT_STORAGE_EQUALS(expectedIndices, expectedValues, tensor);
}

IndexVar i("i"), j("j"), m("m"), n("n"), k("k"), l("l");

IndexArray dlab_indices() {
  IndexArray indices;
  for (int i = 0; i < 10000; ++i) {
    if (i % 2 == 0 || i % 3 == 0) {
      indices.push_back(i);
    }
  }
  return indices;
}

std::vector<double> dlab_values() {
  std::vector<double> values;
  for (int i = 0; i < 10000; ++i) {
    if (i % 2 == 0 || i % 3 == 0) {
      values.push_back(i * ((double)(i % 2 == 0) + (double)(i % 3 == 0)));
    }
  }
  return values;
}

INSTANTIATE_TEST_CASE_P(vector_add, alloc,
    Values(
           TestData(Tensor<double>("a",{10000},Format({Sparse})),
                    32,
                    false,
                    {i},
                    dla("b",Format({Sparse}))(i) +
                    dlb("c",Format({Sparse}))(i),
                    {
                      {
                        // Sparse index
                        {0,6667},
                        dlab_indices()
                      }
                    },
                    dlab_values()
           ),
           TestData(Tensor<double>("a",{10000},Format({Sparse})),
                    32,
                    true,
                    {i},
                    dla("b",Format({Sparse}))(i) +
                    dlb("c",Format({Sparse}))(i),
                    {
                      {
                        // Sparse index
                        {0,6667},
                        dlab_indices()
                      }
                    },
                    dlab_values()
           )
//           ,
//           TestData(Tensor<double>("a",{10000},Format({Fixed}),32),
//                    {i},
//                    dla("b",Format({Fixed}))(i) +
//                    dlb("c",Format({Fixed}))(i),
//                    {
//                        {
//                            // Fixed index
//                            {6667},
//                            dlab_indices()
//                        }
//                    },
//                    dlab_values()
//           )
    )
);

}

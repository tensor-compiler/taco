#include "test.h"
#include "test_tensors.h"

#include "tensor.h"
#include "var.h"
#include "expr.h"
#include "expr_nodes.h"
#include "storage/storage.h"
#include "operator.h"

using namespace taco;

namespace storage_alloc_tests {

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

struct alloc : public TestWithParam<TestData> {};

TEST_P(alloc, storage) {
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
  iassert(expectedIndices.size() == levels.size());
  auto size = storage.getSize();

  for (size_t i=0; i < levels.size(); ++i) {
    auto expectedIndex = expectedIndices[i];
    auto levelIndex = storage.getLevelIndex(i);
    auto levelIndexSize = size.levelIndices[i];

    switch (levels[i].getType()) {
      case LevelType::Dense: {
        iassert(expectedIndex.size() == 1) << "Dense indices have a ptr array";
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
      case LevelType::Fixed: {
        iassert(expectedIndex.size() == 2);
        ASSERT_ARRAY_EQ(expectedIndex[0], {levelIndex.ptr, 1});
        ASSERT_ARRAY_EQ(expectedIndex[1], {levelIndex.idx, levelIndexSize.idx});
        ASSERT_EQ((size_t)1, levelIndexSize.ptr);
        break;
      }
      case LevelType::Offset:
      case LevelType::Replicated:
        break;
    }
  }

  auto& expectedValues = GetParam().expectedValues;
  ASSERT_EQ(expectedValues.size(), storage.getSize().values);
  ASSERT_ARRAY_EQ(expectedValues, {storage.getValues(), size.values});
}

Var i("i"), j("j"), m("m"), n("n");
Var k("k", Var::Sum), l("l", Var::Sum);

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
           TestData(Tensor<double>("a",{10000},Format({Sparse}),32),
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
           )
);

}

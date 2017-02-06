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

typedef int                     IndexType;
typedef std::vector<IndexType>  IndexArray; // Index values
typedef std::vector<IndexArray> Index;      // [0,2] index arrays per Index
typedef std::vector<Index>      Indices;    // One Index per level

struct APIStorage {
	APIStorage(Tensor<double> tensor,
           const Indices& expectedIndices,
           const vector<double> expectedValues)
      : tensor(tensor),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
  }

  Tensor<double> tensor;
  Indices        expectedIndices;
  vector<double> expectedValues;
};

struct api : public TestWithParam<APIStorage> {};

TEST_P(api, pack) {
  Tensor<double> tensor = GetParam().tensor;

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


INSTANTIATE_TEST_CASE_P(load, api,
  Values(
		APIStorage(d33a_CSR("A"),
		{
		  {
			// Dense index
			{3}
		  },
		  {
			// Sparse index
			{0, 1, 1, 3},
			{1, 0, 2},
		  }
		},
		{2, 3, 4}
		)
  )
);


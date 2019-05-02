#include "test.h"
#include "test_tensors.h"

#include <tuple>

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/storage/storage.h"
#include "taco/util/strings.h"

using namespace taco;

typedef std::tuple<std::vector<TensorData<double>>,
                   std::vector<ModeFormat>,
                   std::vector<int>> TestData;

struct format : public TestWithParam<TestData> {};

TEST_P(format, pack) {
  const TensorData<double>& data = std::get<0>(GetParam())[0];

  Format format(std::get<1>(GetParam()), std::get<2>(GetParam()));
  Tensor<double> tensor = data.makeTensor("tensor", format);
  tensor.pack();

  EXPECT_TRUE(data.compare(tensor));
}

template <class ...Ts>
std::vector<TensorData<double>> packageInputs(Ts... inputs) {
  return {inputs...};
}

const auto modeTypes1 = generateModeTypes(1);
const auto modeTypes2 = generateModeTypes(2);
const auto modeTypes3 = generateModeTypes(3);

const auto modeOrderings1 = generateModeOrderings(1);
const auto modeOrderings2 = generateModeOrderings(2);
const auto modeOrderings3 = generateModeOrderings(3);

INSTANTIATE_TEST_CASE_P(vector, format, Combine(
    Values(
        packageInputs(d1a_data()),
        packageInputs(d1b_data()),
        packageInputs(d5a_data()),
        packageInputs(d5b_data()),
        packageInputs(d5c_data())
    ), ValuesIn(modeTypes1), ValuesIn(modeOrderings1)));

INSTANTIATE_TEST_CASE_P(matrix, format, Combine(
    Values(
        packageInputs(d33a_data()),
        packageInputs(d33b_data())
    ), ValuesIn(modeTypes2), ValuesIn(modeOrderings2)));

INSTANTIATE_TEST_CASE_P(tensor3, format, Combine(
    Values(
        packageInputs(d233a_data()),
        packageInputs(d233b_data())
    ), ValuesIn(modeTypes3), ValuesIn(modeOrderings3)));

TEST(format, sparse) {
  Tensor<double> A = d33a("A", Sparse);
  A.pack();
  ASSERT_COMPONENTS_EQUALS({{{0,2}, {0,2}}, {{0,1,3}, {1,0,2}}}, {2,3,4}, A);
}

TEST(format, dense) {
  Tensor<double> A = d33a("A", Dense);
  A.pack();
  ASSERT_COMPONENTS_EQUALS({{{3}}, {{3}}}, {0,2,0, 0,0,0, 3,0,4}, A);
}

TEST(format, block) {
  ModeFormat modeFormat = Dense;
  ASSERT_FALSE(modeFormat.hasFixedSize());
  ASSERT_TRUE(modeFormat.defined());
  modeFormat = Dense(12);
  ASSERT_TRUE(modeFormat.hasFixedSize());
  ASSERT_EQ(12, modeFormat.size());
  ASSERT_TRUE(modeFormat.defined());


  Format format({{Dense(5), Sparse}, {Dense, Dense(5)}});
  ASSERT_TRUE(format.isBlocked());
  ASSERT_EQ(2, format.numBlockLevels());

  std::vector<std::vector<int>> expectedBlockSizes({{5,0}, {0,5}});
  std::vector<int> expectedDimensionFreeSizeBlock({1,0});

  for (int block = 0; block < 2; block++) {
    for (int dim = 0; dim < 2; dim++) {
      ASSERT_EQ(expectedBlockSizes[block][dim], format.getBlockSizes()[block][dim]);
    }
  }
  for (int dim = 0; dim < 2; dim++) {
    ASSERT_EQ(expectedDimensionFreeSizeBlock[dim], format.getDimensionFreeSizeBlock()[dim]);
  }
}

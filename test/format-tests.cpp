#include "test.h"
#include "test_tensors.h"

#include <tuple>

#include "taco/tensor.h"
#include "taco/expr.h"
#include "taco/format.h"
#include "taco/storage/storage.h"
#include "taco/util/strings.h"

using namespace taco;

typedef std::tuple<std::vector<TensorData<double>>,
                   std::vector<ModeType>,
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

const auto dimTypes1 = generateModeTypes(1);
const auto dimTypes2 = generateModeTypes(2);
const auto dimTypes3 = generateModeTypes(3);

const auto dimOrders1 = generateDimensionOrders(1);
const auto dimOrders2 = generateDimensionOrders(2);
const auto dimOrders3 = generateDimensionOrders(3);

INSTANTIATE_TEST_CASE_P(vector, format, Combine(
    Values(
        packageInputs(d1a_data()),
        packageInputs(d1b_data()),
        packageInputs(d5a_data()),
        packageInputs(d5b_data()),
        packageInputs(d5c_data())
    ), ValuesIn(dimTypes1), ValuesIn(dimOrders1)));

INSTANTIATE_TEST_CASE_P(matrix, format, Combine(
    Values(
        packageInputs(d33a_data()),
        packageInputs(d33b_data())
    ), ValuesIn(dimTypes2), ValuesIn(dimOrders2)));

INSTANTIATE_TEST_CASE_P(tensor3, format, Combine(
    Values(
        packageInputs(d233a_data()),
        packageInputs(d233b_data())
    ), ValuesIn(dimTypes3), ValuesIn(dimOrders3)));

TEST(format, sparse) {
  Tensor<double> A = d33a("A", Sparse);
  A.pack();
  ASSERT_STORAGE_EQUALS({{{0,2}, {0,2}}, {{0,1,3}, {1,0,2}}}, {2,3,4}, A);
}

TEST(format, dense) {
  Tensor<double> A = d33a("A", Dense);
  A.pack();
  ASSERT_STORAGE_EQUALS({{{3}}, {{3}}}, {0,2,0, 0,0,0, 3,0,4}, A);
}

#include "test.h"
#include "test_tensors.h"
#include "expr-tests.h"

#include <tuple>

#include "tensor.h"
#include "expr.h"
#include "format.h"
#include "packed_tensor.h"
#include "util/strings.h"

using namespace taco;

typedef std::tuple<std::vector<TensorData<double>>, 
                   Format::LevelTypes,
                   Format::DimensionOrders> TestData;

struct format : public TestWithParam<TestData> {};

TEST_P(format, pack) {
  const TensorData<double>& data = std::get<0>(GetParam())[0];

  Format format(std::get<1>(GetParam()), std::get<2>(GetParam()));
  Tensor<double> tensor = data.makeTensor("tensor", format);

  std::cout << tensor << std::endl;

  EXPECT_TRUE(data.compare(tensor));
}

template <class ...Ts>
std::vector<TensorData<double>> packageInputs(Ts... inputs) {
  return {inputs...};
}

const auto levels1 = generateLevels(1);
const auto levels2 = generateLevels(2);
const auto levels3 = generateLevels(3);

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
    ), ValuesIn(levels1), ValuesIn(dimOrders1)));

INSTANTIATE_TEST_CASE_P(matrix, format, Combine(
    Values(
        packageInputs(d33a_data()),
        packageInputs(d33b_data())
    ), ValuesIn(levels2), ValuesIn(dimOrders2)));

INSTANTIATE_TEST_CASE_P(tensor3, format, Combine(
    Values(
        packageInputs(d233a_data()),
        packageInputs(d233b_data())
    ), ValuesIn(levels3), ValuesIn(dimOrders3)));

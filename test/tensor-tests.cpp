#include "test.h"
#include "taco/tensor.h"

#include <vector>
#include "taco/util/collections.h"

using namespace taco;

TEST(tensor, double_scalar) {
  Tensor<double> a(4.2);
  ASSERT_DOUBLE_EQ(4.2, a.begin()->second);
}

TEST(tensor, double_vector) {
  map<vector<int>, double> vals = {{{0}, 1.0},
                                   {{2}, 2.0}};

  Tensor<double> a({5}, Sparse);
  ASSERT_EQ(ComponentType::Double, a.getComponentType());
  ASSERT_EQ(1u, a.getDimensions().size());
  ASSERT_EQ(5,  a.getDimensions()[0]);

  for (auto& val : vals) {
    a.insert(val.first, val.second);
  }
  a.pack();

  for (auto& val : a) {
    ASSERT_TRUE(util::contains(vals, val.first));
    ASSERT_EQ(vals.at(val.first), val.second);
  }

  TensorBase abase = a;
  for (auto& val : iterate<double>(abase)) {
    ASSERT_TRUE(util::contains(vals, val.first));
    ASSERT_EQ(vals.at(val.first), val.second);
  }
}

#include "test.h"
#include "taco/tensor_base.h"

#include <vector>
#include "taco/util/collections.h"

using namespace taco;

TEST(tensor_base, double_type) {
  map<vector<int>, double> vals = {{{0}, 1.0},
                                   {{2}, 2.0}};

  TensorBase a(typeOf<double>(), {5}, SVEC);
  ASSERT_EQ(ComponentType::Double, a.getComponentType());
  ASSERT_EQ(1u, a.getDimensions().size());
  ASSERT_EQ(5,  a.getDimensions()[0]);

  for (auto& val : vals) {
    a.insert(val.first, val.second);
  }
  a.pack();

  // TODO: Iterate over values in some way, e.g.:
//  for (auto& val : a.doubleValues()) {
//    ASSERT_TRUE(util::contains(vals, val.first));
//    ASSERT_EQ(vals.at(val.first), val.second);
//  }
}

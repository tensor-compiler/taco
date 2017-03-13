#include "test.h"
#include "taco/tensor.h"

#include <vector>
#include "taco/util/collections.h"

using namespace taco;

TEST(tensor, double_type) {
  map<vector<int>, double> vals = {{{0}, 1.0},
                                   {{2}, 2.0}};

  Tensor<double> a({5}, SVEC);

  for (auto& val : vals) {
    a.insert(val.first, val.second);
  }
  a.pack();

  for (auto& val : a) {
    ASSERT_TRUE(util::contains(vals, val.first));
    ASSERT_EQ(vals.at(val.first), val.second);
  }
}

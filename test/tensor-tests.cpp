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
  Tensor<double> a({5}, Sparse);
  ASSERT_EQ(Float(64), a.getComponentType());
  ASSERT_EQ(1u, a.getOrder());
  ASSERT_EQ(5,  a.getDimension(0));

  map<vector<int>,double> vals = {{{0}, 1.0}, {{2}, 2.0}};
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

TEST(tensor, iterate) {
  Tensor<double> a({5}, Sparse);
  a.insert({1}, 10.0);
  a.pack();
  ASSERT_TRUE(a.begin() != a.end());
  ASSERT_TRUE(++a.begin() == a.end());
  ASSERT_DOUBLE_EQ(10.0, (a.begin()++)->second);
}

TEST(tensor, iterate_empty) {
  Tensor<double> a({5}, Sparse);
  a.pack();
  ASSERT_TRUE(a.begin() == a.end());
}

TEST(tensor, duplicates) {
  Tensor<double> a({5,5}, Sparse);
  a.insert({1,2}, 42.0);
  a.insert({2,2}, 10.0);
  a.insert({1,2}, 1.0);
  a.pack();
  map<vector<int>,double> vals = {{{1,2}, 43.0}, {{2,2}, 10.0}};
  for (auto& val : a) {
    ASSERT_TRUE(util::contains(vals, val.first));
    ASSERT_EQ(vals.at(val.first), val.second);
  }
}

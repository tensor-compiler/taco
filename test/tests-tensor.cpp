#include "test.h"
#include "taco/tensor.h"
#include "test_tensors.h"

#include <vector>
#include "taco/util/collections.h"

using namespace taco;

TEST(tensor, double_scalar) {
  Tensor<double> a(4.2);
  ASSERT_DOUBLE_EQ(4.2, a.begin()->second);
}

TEST(tensor, double_vector) {
  Tensor<double> a({5}, Sparse);
  ASSERT_EQ(Float64, a.getComponentType());
  ASSERT_EQ(1, a.getOrder());
  ASSERT_EQ(5, a.getDimension(0));

  map<vector<int>,double> vals = {{{0}, 1.0}, {{2}, 2.0}};
  for (auto& val : vals) {
    a.insert(val.first, val.second);
  }
  a.pack();

  for (auto val = a.beginTyped<int>(); val != a.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }

  TensorBase abase = a;
  Tensor<double> abaseIter = iterate<double>(abase);
  for (auto val = abaseIter.beginTyped<int>(); val != abaseIter.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }
}

TEST(tensor, iterate) {
  Tensor<double> a({5}, Sparse);
  a.insert({1}, 10.0);
  a.pack();
  ASSERT_TRUE(a.begin() != a.end());
  ASSERT_TRUE(++a.begin() == a.end());
  ASSERT_DOUBLE_EQ(10.0, a.begin()->second);
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
  for (auto val = a.beginTyped<int>(); val != a.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }
}

TEST(tensor, duplicates_scalar) {
  Tensor<double> a;
  a.insert({}, 1.0);
  a.insert({}, 2.0);
  a.pack();
  auto val = a.begin();
  ASSERT_EQ(val->second, 3.0);
  ASSERT_TRUE(++val == a.end());
}

TEST(tensor, transpose) {
  TensorData<double> testData = TensorData<double>({5, 3, 2}, {
    {{0,0,0}, 0.0},
    {{0,0,1}, 1.0},
    {{0,1,0}, 2.0},
    {{0,1,1}, 3.0},
    {{2,0,0}, 4.0},
    {{2,0,1}, 5.0},
    {{4,0,0}, 6.0},
  });
  TensorData<double> transposedTestData = TensorData<double>({2, 5, 3}, {
    {{0,0,0}, 0.0},
    {{1,0,0}, 1.0},
    {{0,0,1}, 2.0},
    {{1,0,1}, 3.0},
    {{0,2,0}, 4.0},
    {{1,2,0}, 5.0},
    {{0,4,0}, 6.0},
  });

  Tensor<double> tensor = testData.makeTensor("a", Format({Sparse, Dense, Sparse}, {1, 0, 2}));
  tensor.pack();
  Tensor<double> transposedTensor = transposedTestData.makeTensor("b", Format({Sparse, Dense, Sparse}, {1, 0, 2}));
  transposedTensor.pack();
  ASSERT_TRUE(equals(tensor.transpose({2,0,1}), transposedTensor));

  Tensor<double> transposedTensor2 = transposedTestData.makeTensor("b", Format({Sparse, Sparse, Dense}, {2, 1, 0}));
  transposedTensor2.pack();
  ASSERT_TRUE(equals(tensor.transpose({2,0,1}, Format({Sparse, Sparse, Dense}, {2, 1, 0})), transposedTensor2));
  ASSERT_TRUE(equals(tensor.transpose({0,1,2}), tensor));
}

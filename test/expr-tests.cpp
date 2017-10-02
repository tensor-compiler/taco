#include "test.h"
#include "test_tensors.h"

#include "taco/tensor.h"

using namespace taco;

const IndexVar i("i"), j("j"), k("k");

TEST(expr, repeated_operand) {
  Tensor<double> a("a", {8}, Format({Sparse}, {0}));
  Tensor<double> b = d8b("b", Format({Sparse}, {0}));
  b.pack();
  a(i) = b(i) + b(i);
  a.evaluate();

  Tensor<double> expected("a", {8}, Format({Sparse}, {0}));
  expected.insert({0}, 20.0);
  expected.insert({2}, 40.0);
  expected.insert({3}, 60.0);
  expected.pack();
  ASSERT_TRUE(equals(expected,a));
}

TEST(expr, accumulate) {
  Tensor<double> a = d8a("a2", Format({Dense}, {0}));
  Tensor<double> b = d8b("b", Format({Sparse}, {0}));
  a.pack();
  b.pack();
  a(i) += b(i);
  a.evaluate();

  Tensor<double> expected("e", {8}, Format({Dense}, {0}));
  expected.insert({0}, 11.0);
  expected.insert({1}, 2.0);
  expected.insert({2}, 23.0);
  expected.insert({3}, 30.0);
  expected.insert({5}, 4.0);
  expected.pack();
  ASSERT_TRUE(equals(expected,a)) << endl << expected << endl << endl << a;
}

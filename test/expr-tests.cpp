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

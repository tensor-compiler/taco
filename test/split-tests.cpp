#include "test.h"
#include "test_tensors.h"

#include "taco/tensor.h"

using namespace taco;

TEST(split, elmul) {
  TensorBase a("a", Float(64), {8}, Sparse);
  TensorBase b = d8a("b", Sparse);
  TensorBase c = d8b("c", Sparse);
  b.pack();
  c.pack();

  IndexVar i;
  IndexExpr mul = b(i) * c(i);
  a(i) = mul;

  i.split(mul);

  a.evaluate();

  Tensor<double> e("e", {8}, Sparse);
  e.insert({0}, 10.0);
  e.insert({2}, 60.0);
  e.pack();
  ASSERT_TENSOR_EQ(e,a);
}

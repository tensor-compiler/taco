#include "test.h"
#include "test_tensors.h"

#include "tensor.h"
#include "expr.h"
#include "operator.h"

TEST(elwise, add_vector_to_self_dense) {
  Var i("i");

  auto b = vectord5a("d");

  Tensor<double> a({5}, "d");
  a(i) = b(i) + b(i);

  //
}

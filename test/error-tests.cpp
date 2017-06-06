#include "test.h"
#include "test_tensors.h"

#include "taco/tensor.h"
#include "error_messages.h"

using namespace taco;

TEST(error, compute_without_compile) {
  Tensor<double> a({5}, Sparse);
  Tensor<double> b({5}, Sparse);

  IndexVar i;
  a(i) = b(i);

  ASSERT_DEATH(a.compute(), error::compute_without_compile);
}

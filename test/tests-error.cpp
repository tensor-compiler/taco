#include "test.h"
#include "test_tensors.h"

#include "taco/tensor.h"
#include "taco/error/error_messages.h"

using namespace taco;

const IndexVar i("i"), j("j"), k("k");

TEST(error, expr_dimension_mismatch_freevar) {
  Tensor<double> a({3}, Format({Sparse}, {0}));
  Tensor<double> b({5}, Format({Sparse}, {0}));
  ASSERT_DEATH(a(i) = b(i), error::expr_dimension_mismatch);
}

TEST(error, expr_dimension_mismatch_sumvar) {
  Tensor<double> a({5}, Format({Sparse}, {0}));
  Tensor<double> B({5,4}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> c({3}, Format({Sparse}, {0}));
  ASSERT_DEATH(a(i) = B(i,j)*c(j), error::expr_dimension_mismatch);
}

TEST(error, compile_without_expr) {
  Tensor<double> a({5}, Sparse);
  ASSERT_DEATH(a.compile(), error::compile_without_expr);
}

TEST(error, assemble_without_compile) {
  Tensor<double> a({5}, Sparse);
  Tensor<double> b({5}, Sparse);
  a(i) = b(i);
  ASSERT_DEATH(a.assemble(), error::assemble_without_compile);
}

TEST(error, compute_without_compile) {
  Tensor<double> a({5}, Sparse);
  Tensor<double> b({5}, Sparse);
  a(i) = b(i);
  ASSERT_DEATH(a.compute(), error::compute_without_compile);
}

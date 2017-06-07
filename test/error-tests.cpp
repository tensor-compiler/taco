#include "test.h"
#include "test_tensors.h"

#include "taco/tensor.h"
#include "error/error_messages.h"

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

TEST(error, expr_transpose1) {
  Tensor<double> A({5,5}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> B({5,5}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> C({5,5}, Format({Sparse,Sparse}, {1,0}));
  ASSERT_DEATH(A(i,j) = B(i,j) + C(i,j), error::expr_transposition);
}

TEST(error, expr_transpose2) {
  Tensor<double> A({5,5}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> B({5,5}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> C({5,5}, Format({Sparse,Sparse}, {0,1}));
  ASSERT_DEATH(A(i,j) = B(i,j) + C(j,i), error::expr_transposition);
}

TEST(error, expr_transpose3) {
  Tensor<double> A({5,5,5}, Format({Sparse,Sparse,Sparse}, {0,1,2}));
  Tensor<double> B({5,5,5}, Format({Sparse,Sparse,Sparse}, {0,1,2}));
  Tensor<double> C({5,5,5}, Format({Sparse,Sparse,Sparse}, {0,1,2}));
  ASSERT_DEATH(A(i,j,k) = B(i,j,k) + C(k,i,j), error::expr_transposition);
}

TEST(error, expr_distribute) {
  Tensor<double> A({5,5}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> b({5}, Format({Sparse}, {0}));
  ASSERT_DEATH(A(i,j) = b(i), error::expr_distribution);
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

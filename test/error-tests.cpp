#include "test.h"
#include "test_tensors.h"

#include "taco/tensor.h"
#include "error/error_messages.h"

using namespace taco;

const IndexVar i("i"), j("j"), k("k");

TEST(error, compile_without_expr) {
  Tensor<double> a({5}, Sparse);
  ASSERT_DEATH(a.compile(), error::compile_without_expr);
}

TEST(error, transpose1) {
  Tensor<double> A({5,5}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> B({5,5}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> C({5,5}, Format({Sparse,Sparse}, {1,0}));
  A(i,j) = B(i,j) + C(i,j);
  ASSERT_DEATH(A.compile(), error::compile_transposition);
}

TEST(error, transpose2) {
  Tensor<double> A({5,5}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> B({5,5}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> C({5,5}, Format({Sparse,Sparse}, {0,1}));
  A(i,j) = B(i,j) + C(j,i);
  ASSERT_DEATH(A.compile(), error::compile_transposition);
}

TEST(error, transpose3) {
  Tensor<double> A({5,5,5}, Format({Sparse,Sparse,Sparse}, {0,1,2}));
  Tensor<double> B({5,5,5}, Format({Sparse,Sparse,Sparse}, {0,1,2}));
  Tensor<double> C({5,5,5}, Format({Sparse,Sparse,Sparse}, {0,1,2}));
  A(i,j,k) = B(i,j,k) + C(k,i,j);
  ASSERT_DEATH(A.compile(), error::compile_transposition);
}

TEST(error, compute_without_compile) {
  Tensor<double> a({5}, Sparse);
  Tensor<double> b({5}, Sparse);
  a(i) = b(i);
  ASSERT_DEATH(a.compute(), error::compute_without_compile);
}

#include "test.h"
#include "test_tensors.h"

#include "taco/tensor.h"
#include "taco/error/error_messages.h"

using namespace taco;

const IndexVar i("i"), j("j"), k("k");

TEST(error, expr_dimension_mismatch_freevar) {
  Tensor<double> a({3}, Format({Sparse}, {0}));
  Tensor<double> b({5}, Format({Sparse}, {0}));
  ASSERT_THROW(a(i) = b(i), taco::TacoException);
}

TEST(error, expr_dimension_mismatch_sumvar) {
  Tensor<double> a({5}, Format({Sparse}, {0}));
  Tensor<double> B({5,4}, Format({Sparse,Sparse}, {0,1}));
  Tensor<double> c({3}, Format({Sparse}, {0}));
  ASSERT_THROW(a(i) = B(i,j)*c(j), taco::TacoException);
}

TEST(error, compile_without_expr) {
  Tensor<double> a({5}, Sparse);
  ASSERT_THROW(a.compile(), taco::TacoException);
}

TEST(error, compile_tensor_name_collision) {
  Tensor<double> a("a", {5}, Sparse);
  Tensor<double> b("a", {5}, Sparse); // name should be "b"
  a(i) = b(i);
  ASSERT_THROW(a.compile(), taco::TacoException);
}

TEST(error, assemble_without_compile) {
  Tensor<double> a({5}, Sparse);
  Tensor<double> b({5}, Sparse);
  a(i) = b(i);
  ASSERT_THROW(a.assemble(), taco::TacoException);
}

TEST(error, compute_without_compile) {
  Tensor<double> a({5}, Sparse);
  Tensor<double> b({5}, Sparse);
  a(i) = b(i);
  ASSERT_THROW(a.compute(), taco::TacoException);
}

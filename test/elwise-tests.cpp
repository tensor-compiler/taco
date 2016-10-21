#include "test.h"
#include "test_tensors.h"

#include "tensor.h"
#include "var.h"
#include "expr.h"
#include "packed_tensor.h"
#include "operator.h"

TEST(elwise, vector_neg_dense) {
  Var i("i");

  Tensor<double> a({5}, "d");
  Tensor<double> b = vectord5a("d");
  b.pack();

  a(i) = -b(i);

  a.compile();
  a.assemble();
  a.evaluate();

  auto apack = a.getPackedTensor();
  ASSERT_NE(nullptr, apack);

  auto& indices = apack->getIndices();
  auto& values  = apack->getValues();

  ASSERT_EQ(1u, indices.size());
  ASSERT_EQ(0u, indices[0].size());

  ASSERT_VECTOR_EQ({0.0, -1.0, 0.0, 0.0, -2.0}, values);
}

TEST(elwise, matrix_neg_dense) {
  Var i("i");
  Var j("j");

  Tensor<double> A({3, 3}, "dd");
  Tensor<double> B = matrixd33a("dd");
  B.pack();

  A(i,j) = -B(i,j);

  A.compile();
  A.assemble();
  A.evaluate();

  auto Apack = A.getPackedTensor();
  ASSERT_NE(nullptr, Apack);

  auto& indices = Apack->getIndices();
  auto& values  = Apack->getValues();

  ASSERT_EQ(2u, indices.size());
  ASSERT_EQ(0u, indices[1].size());
  ASSERT_EQ(0u, indices[2].size());

  ASSERT_VECTOR_EQ({{ 0, -1,  0,
                      0,  0,  0,
                     -2,  0, -3}}, values);
}

TEST(elwise, vector_add_dense) {
  Var i("i");

  Tensor<double> a({5}, "d");
  Tensor<double> b = vectord5a("d");
  Tensor<double> c = vectord5a("d");
  b.pack();
  c.pack();

  a(i) = b(i) + c(i);

  a.compile();
  a.assemble();
  a.evaluate();

  auto apack = a.getPackedTensor();
  ASSERT_NE(nullptr, apack);

  auto& indices = apack->getIndices();
  auto& values  = apack->getValues();

  ASSERT_EQ(1u, indices.size());
  ASSERT_EQ(0u, indices[0].size());

  ASSERT_VECTOR_EQ({0.0, 2.0, 0.0, 0.0, 4.0}, values);
}

TEST(elwise, matrix_add_dense) {
  Var i("i");
  Var j("j");

  Tensor<double> A({3, 3}, "dd");
  Tensor<double> B = matrixd33a("dd");
  Tensor<double> C = matrixd33a("dd");
  B.pack();
  C.pack();

  A(i,j) = B(i,j) + C(i,j);

  A.compile();
  A.assemble();
  A.evaluate();

  auto Apack = A.getPackedTensor();
  ASSERT_NE(nullptr, Apack);

  auto& indices = Apack->getIndices();
  auto& values  = Apack->getValues();

  ASSERT_EQ(2u, indices.size());
  ASSERT_EQ(0u, indices[1].size());
  ASSERT_EQ(0u, indices[2].size());

  ASSERT_VECTOR_EQ({{0, 2, 0,
                     0, 0, 0,
                     4, 0, 6}}, values);
}

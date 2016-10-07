#include "test.h"
#include "test_tensors.h"

#include "tensor.h"
#include "expr.h"
#include "operator.h"

TEST(elwise, add_vector_to_self_dense) {
  Var i("i");

  Tensor<double> a({5}, "d");
  Tensor<double> b = vectord5a("d");
  b.pack();

  a(i) = b(i) + b(i);

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

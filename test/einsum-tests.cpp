#include "test.h"
#include "taco/expr/expr.h"

using namespace taco;

Type vectorType(Float64, {3});
Type matrixType(Float64, {3,3});

const IndexVar i("i"), j("j"), k("k");

TEST(einsum, verify) {
  TensorVar b("b", Float64), c("c", Float64), d("d", Float64), e("e", Float64);

  ASSERT_TRUE(verifyEinsum(b + c));
  ASSERT_TRUE(verifyEinsum(b*c*d*e));
  ASSERT_TRUE(verifyEinsum(b*c + d*e));
  ASSERT_TRUE(verifyEinsum(b*c - d*e));

  ASSERT_FALSE(verifyEinsum(b/c));
  ASSERT_FALSE(verifyEinsum(b*(c+d)));
}

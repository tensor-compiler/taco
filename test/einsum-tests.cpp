#include "test.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;

Type vectorType(Float64, {3});
Type matrixType(Float64, {3,3});

const IndexVar i("i"), j("j"), k("k");

TEST(einsum, verify) {
  TensorVar a("a", Float64), b("b", Float64), c("c", Float64), d("d", Float64),
            e("e", Float64);

  ASSERT_TRUE(isEinsumNotation(a = b + c));
  ASSERT_TRUE(isEinsumNotation(a = b*c*d*e));
  ASSERT_TRUE(isEinsumNotation(a = b*c + d*e));
  ASSERT_TRUE(isEinsumNotation(a = b*c - d*e));

  ASSERT_FALSE(isEinsumNotation(a = b/c));
  ASSERT_FALSE(isEinsumNotation(a = b*(c+d)));
}

TEST(einsum, scalars) {
  TensorVar a("a", Float64), b("b", Float64), c("c", Float64), d("d", Float64);

  ASSERT_TRUE(equals(makeReductionNotation(a = b*c),   a = b*c));
  ASSERT_TRUE(equals(makeReductionNotation(a = b*c*d), a = b*c*d));
  ASSERT_TRUE(equals(makeReductionNotation(a = b+d),   a = b+d));
  ASSERT_TRUE(equals(makeReductionNotation(a = b-d),   a = b-d));
}

TEST(einsum, vectors) {
  TensorVar a("a", Float64), b("b", vectorType), c("c", vectorType),
            d("d", vectorType), e("e", vectorType), f("f", vectorType);

  ASSERT_TRUE(equals(makeReductionNotation(a=b(i)*c(i)),
                     a = sum(i, b(i)*c(i))));
  ASSERT_TRUE(equals(makeReductionNotation(a=b(i)*c(i)*d(i)),
                     a = sum(i, b(i)*c(i)*d(i))));
  ASSERT_TRUE(equals(makeReductionNotation(a=b(i)*c(j)),
                     a = sum(i, sum(j, b(i)*c(j)))));
  ASSERT_TRUE(equals(makeReductionNotation(a=b(i)*c(j)*d(k)),
                     a = sum(i, sum(j, sum(k, b(i)*c(j)*d(k))))));

  ASSERT_TRUE(equals(makeReductionNotation(a=b(i)+c(j)),
                     a = sum(i, b(i)) + sum(j, c(j))));
  ASSERT_TRUE(equals(makeReductionNotation(a=b(i)+c(i)),
                     a = sum(i, b(i)) + sum(i, c(i))));

  ASSERT_TRUE(equals(makeReductionNotation(a = b(i)*c(i) + d(j)*e(j)),
                     a = sum(i, b(i)*c(i)) + sum(j, d(j)*e(j))));

  ASSERT_TRUE(equals(makeReductionNotation(a = b(i)*c(i) + d(i)*e(j)),
                     a = sum(i, b(i)*c(i)) + sum(i, sum(j, d(i)*e(j)))));

  ASSERT_TRUE(equals(makeReductionNotation(f(i)=b(i)*c(i)),
                     f(i) = b(i)*c(i)));
}

TEST(einsum, matrices) {
  TensorVar alpha("alpha", Float64);
  TensorVar a("a", vectorType), b("b", vectorType), c("c", vectorType);
  TensorVar B("B", matrixType), C("C", matrixType);

  ASSERT_TRUE(equals(makeReductionNotation(alpha = B(i,j)*C(i,j)),
                     alpha = sum(i, sum(j, B(i,j)*C(i,j) ))));

  ASSERT_TRUE(equals(makeReductionNotation(a(i)=B(i,j)*c(j)),
                     a(i) = sum(j, B(i,j)*c(j))));
}

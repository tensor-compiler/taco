#include "test.h"
#include "taco/expr/expr.h"

using namespace taco;

Type vectorType(Float64, {3});
Type matrixType(Float64, {3,3});

const IndexVar i("i"), j("j"), k("k");

TEST(einsum, verify) {
  TensorVar b("b", Float64), c("c", Float64), d("d", Float64), e("e", Float64);

  ASSERT_TRUE(doesEinsumApply(b + c));
  ASSERT_TRUE(doesEinsumApply(b*c*d*e));
  ASSERT_TRUE(doesEinsumApply(b*c + d*e));
  ASSERT_TRUE(doesEinsumApply(b*c - d*e));

  ASSERT_FALSE(doesEinsumApply(b/c));
  ASSERT_FALSE(doesEinsumApply(b*(c+d)));
}

TEST(einsum, scalars) {
  TensorVar b("b", Float64), c("c", Float64), d("d", Float64);

  ASSERT_TRUE(equals(einsum(b*c), b*c));
  ASSERT_TRUE(equals(einsum(b*c*d), b*c*d));
  ASSERT_TRUE(equals(einsum(b+d), b+d));
  ASSERT_TRUE(equals(einsum(b-d), b-d));
}

TEST(einsum, vectors) {
  TensorVar b("b", vectorType), c("c", vectorType), d("d", vectorType),
            e("e", vectorType);

  ASSERT_TRUE(equals(einsum( b(i)*c(i) ), sum(i)( b(i)*c(i) )));
  ASSERT_TRUE(equals(einsum( b(i)*c(i)*d(i) ), sum(i)( b(i)*c(i)*d(i) )));
  ASSERT_TRUE(equals(einsum( b(i)*c(j) ), sum(i)(sum(j)( b(i)*c(j) ))));
  ASSERT_TRUE(equals(einsum( b(i)*c(j)*d(k) ),
                     sum(i)(sum(j)(sum(k)( b(i)*c(j)*d(k) )))));

  ASSERT_TRUE(equals(einsum( b(i)+c(j) ), sum(i)( b(i) ) + sum(j)( c(j) )));
  ASSERT_TRUE(equals(einsum( b(i)+c(i) ), sum(i)( b(i) ) + sum(i)( c(i) )));

  ASSERT_TRUE(equals(einsum(b(i)*c(i) + d(j)*e(j)),
                     sum(i)(b(i)*c(i)) + sum(j)(d(j)*e(j))));

  ASSERT_TRUE(equals(einsum(b(i)*c(i) + d(i)*e(j)),
                     sum(i)(b(i)*c(i)) + sum(i)(sum(j)(d(i)*e(j)))));

  ASSERT_TRUE(equals(einsum( b(i)*c(i), {i}), b(i)*c(i) ));
}

TEST(einsum, matrices) {
  TensorVar b("b", vectorType), c("c", vectorType);
  TensorVar B("B", matrixType), C("C", matrixType);

  ASSERT_TRUE(equals(einsum( B(i,j)*C(i,j) ), sum(i)(sum(j)( B(i,j)*C(i,j) ))));
  ASSERT_TRUE(equals(einsum( B(i,j)*c(j), {i}), sum(j)( B(i,j)*c(j) )));

}



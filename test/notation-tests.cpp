#include "test.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;

static TensorVar as("a", Float64), bs("b", Float64), cs("c", Float64),
                 ds("d", Float64), es("e", Float64);

static Type vectorType(Float64, {3});
static TensorVar a("a", vectorType), b("b", vectorType), c("c", vectorType),
                 d("d", vectorType), e("e", vectorType), f("f", vectorType);

static Type matrixType(Float64, {3,3});
static TensorVar A("A", matrixType), B("B", matrixType), C("C", matrixType);

static const IndexVar i("i"), j("j"), k("k");


TEST(notation, isEinsumNotation) {
  ASSERT_TRUE(isEinsumNotation(as = bs + cs));
  ASSERT_TRUE(isEinsumNotation(as = bs*cs*ds*es));
  ASSERT_TRUE(isEinsumNotation(as = bs*cs + ds*es));
  ASSERT_TRUE(isEinsumNotation(as = bs*cs - ds*es));
  ASSERT_TRUE(isEinsumNotation(a(i) = b(i) + c(i)));
  ASSERT_TRUE(isEinsumNotation(A(i,j) = B(i,j) + C(i,j)));
  ASSERT_TRUE(isEinsumNotation(a(i) = B(i,j) * c(j)));

  ASSERT_FALSE(isEinsumNotation(as = bs/cs));
  ASSERT_FALSE(isEinsumNotation(as = bs*(cs+ds)));
  ASSERT_FALSE(isEinsumNotation(a(i) = sum(j, B(i,j) * c(j))));
  ASSERT_FALSE(isEinsumNotation(forall(i, a(i)=b(i))));
}

TEST(notation, isReductionNotation) {
  ASSERT_TRUE(isReductionNotation(as = bs + cs));
  ASSERT_TRUE(isReductionNotation(as = bs*cs*ds*es));
  ASSERT_TRUE(isReductionNotation(as = bs*cs + ds*es));
  ASSERT_TRUE(isReductionNotation(as = bs*cs - ds*es));
  ASSERT_TRUE(isReductionNotation(as = bs/cs));
  ASSERT_TRUE(isReductionNotation(as = bs*(cs+ds)));
  ASSERT_TRUE(isReductionNotation(a(i) = b(i) + c(i)));
  ASSERT_TRUE(isReductionNotation(A(i,j) = B(i,j) + C(i,j)));
  ASSERT_TRUE(isReductionNotation(a(i) = sum(j, B(i,j) * c(j))));

  ASSERT_FALSE(isReductionNotation(a(i) = B(i,j) * c(j)));
  ASSERT_FALSE(isReductionNotation(forall(i, a(i)=b(i))));
}

TEST(notation, isConcreteNotation) {
  ASSERT_TRUE(isConcreteNotation(as = bs + cs));
  ASSERT_TRUE(isConcreteNotation(as = bs*cs*ds*es));
  ASSERT_TRUE(isConcreteNotation(as = bs*cs + ds*es));
  ASSERT_TRUE(isConcreteNotation(as = bs*cs - ds*es));
  ASSERT_TRUE(isConcreteNotation(as = bs/cs));
  ASSERT_TRUE(isConcreteNotation(as = bs*(cs+ds)));


  ASSERT_TRUE(isConcreteNotation(forall(i, a(i) = b(i) + c(i))));
  ASSERT_TRUE(isConcreteNotation(forall(i,
                                        forall(j,
                                               A(i,j) = B(i,j) + C(i,j)))));
  ASSERT_TRUE(isConcreteNotation(forall(j,
                                        forall(i,
                                               A(i,j) = B(i,j) + C(i,j)))));
  ASSERT_TRUE(isConcreteNotation(forall(i,
                                        forall(j,
                                               a(i) += B(i,j) * c(j)))));

  ASSERT_FALSE(isConcreteNotation(a(i) = b(i) + c(i)));
  ASSERT_FALSE(isConcreteNotation(A(i,j) = B(i,j) + C(i,j)));
  ASSERT_FALSE(isConcreteNotation(a(i) = B(i,j) * c(j)));
  ASSERT_FALSE(isConcreteNotation(a(i) = sum(j, B(i,j) * c(j))));
  ASSERT_FALSE(isConcreteNotation(forall(i,
                                         A(i,j) = B(i,j) + C(i,j))));
  ASSERT_FALSE(isConcreteNotation(forall(i,
                                         forall(j,
                                                a(i) = B(i,j) * c(j)))));
  ASSERT_FALSE(isConcreteNotation(forall(i,
                                         forall(j,
                                                a(i) = sum(j,
                                                           B(i,j) * c(j))))));
  ASSERT_FALSE(isConcreteNotation(forall(i,
                                         forall(j,
                                                a(i) += sum(j,
                                                            B(i,j) * c(j))))));
}

TEST(notation, makeReductionNotation) {
  ASSERT_TRUE(equals(makeReductionNotation(as = bs*cs),   as = bs*cs));
  ASSERT_TRUE(equals(makeReductionNotation(as = bs*cs*ds), as = bs*cs*ds));
  ASSERT_TRUE(equals(makeReductionNotation(as = bs+ds),   as = bs+ds));
  ASSERT_TRUE(equals(makeReductionNotation(as = bs-ds),   as = bs-ds));

  ASSERT_TRUE(equals(makeReductionNotation(as=b(i)*c(i)),
                     as = sum(i, b(i)*c(i))));
  ASSERT_TRUE(equals(makeReductionNotation(as=b(i)*c(i)*d(i)),
                     as = sum(i, b(i)*c(i)*d(i))));
  ASSERT_TRUE(equals(makeReductionNotation(as=b(i)*c(j)),
                     as = sum(i, sum(j, b(i)*c(j)))));
  ASSERT_TRUE(equals(makeReductionNotation(as=b(i)*c(j)*d(k)),
                     as = sum(i, sum(j, sum(k, b(i)*c(j)*d(k))))));

  ASSERT_TRUE(equals(makeReductionNotation(as=b(i)+c(j)),
                     as = sum(i, b(i)) + sum(j, c(j))));
  ASSERT_TRUE(equals(makeReductionNotation(as=b(i)+c(i)),
                     as = sum(i, b(i)) + sum(i, c(i))));

  ASSERT_TRUE(equals(makeReductionNotation(as = b(i)*c(i) + d(j)*e(j)),
                     as = sum(i, b(i)*c(i)) + sum(j, d(j)*e(j))));

  ASSERT_TRUE(equals(makeReductionNotation(as = b(i)*c(i) + d(i)*e(j)),
                     as = sum(i, b(i)*c(i)) + sum(i, sum(j, d(i)*e(j)))));

  ASSERT_TRUE(equals(makeReductionNotation(f(i)=b(i)*c(i)),
                     f(i) = b(i)*c(i)));

  ASSERT_TRUE(equals(makeReductionNotation(as = B(i,j)*C(i,j)),
                     as = sum(i, sum(j, B(i,j)*C(i,j) ))));

  ASSERT_TRUE(equals(makeReductionNotation(a(i)=B(i,j)*c(j)),
                     a(i) = sum(j, B(i,j)*c(j))));
}


#include "test.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;

static TensorVar alpha("alpha", Float64);
static TensorVar beta("beta",   Float64);

static TensorVar as("a", Float64), bs("b", Float64), cs("c", Float64),
                 ds("d", Float64), es("e", Float64);

static Type vectorType(Float64, {3});
static TensorVar a("a", vectorType), b("b", vectorType), c("c", vectorType),
                 d("d", vectorType), e("e", vectorType), f("f", vectorType);

static Type matrixType(Float64, {3,3});
static TensorVar A("A", matrixType), B("B", matrixType), C("C", matrixType);

static Type tensorType(Float64, {3,3,3});
static TensorVar S("S", tensorType), T("T", tensorType);

static const IndexVar i("i"), j("j"), k("k");

TensorVar ti("ti", Float64);
TensorVar tj("tj", Float64);
TensorVar tk("tk", Float64);

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
  ASSERT_TRUE(isReductionNotation(as = sum(i, sum(j, B(i,j)))));
  ASSERT_TRUE(isReductionNotation(a(i) = sum(j, sum(k, S(i,j,k)))));

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
  ASSERT_NOTATION_EQ(as = bs*cs,    makeReductionNotation(as = bs*cs));
  ASSERT_NOTATION_EQ(as = bs*cs*ds, makeReductionNotation(as = bs*cs*ds));
  ASSERT_NOTATION_EQ(as = bs+ds,    makeReductionNotation(as = bs+ds));
  ASSERT_NOTATION_EQ(as = bs-ds,    makeReductionNotation(as = bs-ds));

  ASSERT_NOTATION_EQ(as = sum(i, b(i)*c(i)),
                     makeReductionNotation(as = b(i)*c(i)));
  ASSERT_NOTATION_EQ(as = sum(i, b(i)*c(i)*d(i)),
                     makeReductionNotation(as=b(i)*c(i)*d(i)));
  ASSERT_NOTATION_EQ(as = sum(i, sum(j, b(i)*c(j))),
                     makeReductionNotation(as=b(i)*c(j)));
  ASSERT_NOTATION_EQ(as = sum(i, sum(j, sum(k, b(i)*c(j)*d(k)))),
                     makeReductionNotation(as=b(i)*c(j)*d(k)));

  ASSERT_NOTATION_EQ(as = sum(i, b(i)) + sum(j, c(j)),
                     makeReductionNotation(as=b(i)+c(j)));
  ASSERT_NOTATION_EQ(as = sum(i, b(i)) + sum(i, c(i)),
                     makeReductionNotation(as=b(i)+c(i)));
  ASSERT_NOTATION_EQ(as = sum(i, b(i)*c(i)) + sum(j, d(j)*e(j)),
                     makeReductionNotation(as = b(i)*c(i) + d(j)*e(j)));
  ASSERT_NOTATION_EQ(as = sum(i, b(i)*c(i)) + sum(i, sum(j, d(i)*e(j))),
                     makeReductionNotation(as = b(i)*c(i) + d(i)*e(j)));
  ASSERT_NOTATION_EQ(f(i) = b(i)*c(i),
                     makeReductionNotation(f(i)=b(i)*c(i)));
  ASSERT_NOTATION_EQ(as = sum(i, sum(j, B(i,j)*C(i,j) )),
                     makeReductionNotation(as = B(i,j)*C(i,j)));
  ASSERT_NOTATION_EQ(a(i) = sum(j, B(i,j)*c(j)),
                     makeReductionNotation(a(i)=B(i,j)*c(j)));
}

struct ConcreteTest {
  ConcreteTest(IndexStmt reduction, IndexStmt concrete)
      : reduction(reduction), concrete(concrete) {}

  IndexStmt reduction;
  IndexStmt concrete;

  friend ostream &operator<<(ostream& os, const ConcreteTest& data) {
    return os << endl << "Reduction:         " << data.reduction
              << endl << "Expected Concrete: " << data.concrete;
  }
};
struct concrete : public TestWithParam<ConcreteTest> {};

TEST_P(concrete, notation) {
  string reason;
  ASSERT_TRUE(isReductionNotation(GetParam().reduction))
      << GetParam().reduction;
  ASSERT_TRUE(isConcreteNotation(GetParam().concrete, &reason))
      << GetParam().concrete << std::endl << reason;
  ASSERT_NOTATION_EQ(GetParam().concrete,
                     makeConcreteNotation(GetParam().reduction));
}

// For scalar expressions nothing changes
INSTANTIATE_TEST_CASE_P(scalar, concrete,
  Values(ConcreteTest(as = bs*cs,
                      as = bs*cs),
         ConcreteTest(as = bs*cs*ds,
                      as = bs*cs*ds),
         ConcreteTest(as = bs+ds,
                      as = bs+ds),
         ConcreteTest(as = bs-ds,
                      as = bs-ds)));

// For element-wise tensor expressions (without reductions) add forall loops
INSTANTIATE_TEST_CASE_P(elwise, concrete,
  Values(ConcreteTest(a(i) = b(i) + c(i),
                      forall(i,
                             a(i) = b(i) + c(i))),
         ConcreteTest(A(i,j) = B(i,j) + C(i,j),
                      forall(i,
                             forall(j,
                                    A(i,j) = B(i,j) + C(i,j))))));

// If the result is a tensor, then introduce a temporary (tj)
INSTANTIATE_TEST_CASE_P(reduce_into_temporary, concrete,
  Values(ConcreteTest(alpha = sum(i, b(i)),
                      forall(i,
                             alpha += b(i))),
         ConcreteTest(a(i) = sum(j, B(i,j)*c(j)),
                      forall(i,
                             forall(j,
                                    a(i) += B(i,j)*c(j)))),
         ConcreteTest(a(i) = sum(j, sum(k, S(i,j,k))),
                      forall(i,
                             forall(j,
                                    forall(k,
                                           a(i) += S(i,j,k)))))));

// separate reductions require separate temporaries
INSTANTIATE_TEST_CASE_P(separate_reductions, concrete,
  Values(ConcreteTest(a(i) = sum(j, B(i,j)) + sum(k, C(i,k)),
                      forall(i, where(where(a(i) = tj + tk,
                                            forall(k,
                                                   tk += C(i,k))),
                                      forall(j, tj += B(i,j))))),
         ConcreteTest(as = sum(j, b(j)) + sum(k, c(k)),
                      where(where(as = tj + tk,
                                  forall(k,
                                         tk += c(k))),
                            forall(j, tj += b(j))))));


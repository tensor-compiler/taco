#include "test.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/index_notation/index_notation.h"
#include "op_factory.h"

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

static Type largeMatrixType(Float64, {4,4});
static TensorVar D("D", largeMatrixType), E("E", largeMatrixType),
                 F("F", largeMatrixType), G("G", largeMatrixType, CSR);

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
  ASSERT_TRUE(isConcreteNotation(suchthat(as = bs + cs, {})));
  ASSERT_FALSE(isConcreteNotation(suchthat(suchthat(as = bs + cs, {}), {})));
  ASSERT_FALSE(isConcreteNotation(forall(i, suchthat(a(i) = b(i) + c(i), {}))));
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

TEST(notation, isomorphic) {
  ASSERT_TRUE(isomorphic(A(i,j) = B(i,j) + C(i,j), A(i,j) = B(i,j) + C(i,j)));
  ASSERT_TRUE(isomorphic(A(i,j) = B(i,j) + C(i,j), B(i,j) = C(i,j) + A(i,j)));
  ASSERT_TRUE(isomorphic(A(i,j) = B(i,j) + C(i,j), A(i,k) = B(i,k) + C(i,k)));
  ASSERT_TRUE(isomorphic(A(i,j) = B(i,j) + C(i,j), A(j,i) = B(j,i) + C(j,i)));
  ASSERT_FALSE(isomorphic(A(i,j) = B(i,j) + C(i,j), A(i,k) = B(i,k) + C(k,i)));
  ASSERT_FALSE(isomorphic(A(i,j) = B(i,j) + C(i,j), A(i,k) = B(i,k) + C(i,j)));
  ASSERT_FALSE(isomorphic(A(i,j) = B(i,j) + C(i,j), D(i,j) = E(i,j) + F(i,j)));
  ASSERT_FALSE(isomorphic(D(i,j) = E(i,j) + F(i,j), D(i,j) = E(i,j) + G(i,j)));
  ASSERT_TRUE(isomorphic(forall(i, forall(j, A(i,j) = B(i,j) + C(i,j))),
                         forall(j, forall(i, A(j,i) = B(j,i) + C(j,i)))));
  ASSERT_FALSE(isomorphic(forall(i, forall(j, A(i,j) = B(i,j) + C(i,j))),
                          forall(i, forall(j, A(j,i) = B(j,i) + C(j,i)))));
  ASSERT_FALSE(isomorphic(forall(i, forall(j, A(i,j) = B(i,j) + C(i,j),
                                 MergeStrategy::TwoFinger, ParallelUnit::DefaultUnit, OutputRaceStrategy::NoRaces)),
                          forall(j, forall(i, A(j,i) = B(j,i) + C(j,i)))));
  ASSERT_TRUE(isomorphic(sum(j, B(i,j) + C(i,j)), sum(i, B(j,i) + C(j,i))));
  ASSERT_FALSE(isomorphic(sum(j, B(i,j) + C(i,j)), sum(j, B(j,i) + C(j,i))));
}

TEST(notation, generatePackCOOStmt) {
  ModeFormat compressedNU = ModeFormat::Compressed(ModeFormat::NOT_UNIQUE);
  ModeFormat singletonNU = ModeFormat::Singleton(ModeFormat::NOT_UNIQUE);

  TensorVar ac("a_COO", vectorType, Format({compressedNU}));
  ASSERT_TRUE(isomorphic(forall(i, a(i) = ac(i)), 
                         generatePackCOOStmt(a, {i}, true)));
  ASSERT_TRUE(isomorphic(forall(i, ac(i) = a(i)), 
                         generatePackCOOStmt(a, {i}, false)));

  TensorVar AC("A_COO", matrixType, Format({compressedNU, singletonNU}));
  ASSERT_TRUE(isomorphic(forall(i, forall(j, A(i, j) = AC(i, j))), 
                         generatePackCOOStmt(A, {i, j}, true)));
  ASSERT_TRUE(isomorphic(forall(i, forall(j, AC(i, j) = A(i, j))), 
                         generatePackCOOStmt(A, {i, j}, false)));
  ASSERT_TRUE(isomorphic(forall(j, forall(i, A(j, i) = AC(j, i))), 
                         generatePackCOOStmt(A, {j, i}, true)));

  TensorVar SC("S_COO", tensorType, Format({compressedNU, singletonNU, singletonNU}));
  ASSERT_TRUE(isomorphic(forall(i, forall(j, forall(k, S(i, j, k) = SC(i, j, k)))), 
                         generatePackCOOStmt(S, {i, j, k}, true)));
  ASSERT_TRUE(isomorphic(forall(i, forall(j, forall(k, SC(i, j, k) = S(i, j, k)))), 
                         generatePackCOOStmt(S, {i, j, k}, false)));
  ASSERT_TRUE(isomorphic(forall(j, forall(k, forall(i, S(j, k, i) = SC(j, k, i)))), 
                         generatePackCOOStmt(S, {j, k, i}, true)));
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



Func scOr("Or", OrImpl(), {Annihilator((bool)1), Identity((bool)0)});
Func scAnd("And", AndImpl(), {Annihilator((bool)0), Identity((bool)0)});

Func bfsMaskOp("bfsMask", BfsLower(), BfsMaskAlg());
INSTANTIATE_TEST_CASE_P(tensorOpConcrete, concrete,
              Values(ConcreteTest(a(i) = Reduction(scOr(), j, bfsMaskOp(scAnd(B(i, j), c(j)), c(i))),
                                  forall(i,
                                         forall(j,
                                                Assignment(a(i), bfsMaskOp(scAnd(B(i, j), c(j)), c(i)), scOr())
                                         )))));

// funcIsomorphic ensures that the isomorphic function can proceed without error
// on IndexExpr's that contain `Func`'s.
TEST(notation, funcIsomorphic) {
  int dim = 10;
  Func xorOp("xor", GeneralAdd(), xorGen());
  Tensor<int> A("A", {dim});
  Tensor<int> B("B", {dim});
  IndexVar i;
  auto indexExpr = xorOp(A(i), B(i));
  ASSERT_TRUE(isomorphic(indexExpr, indexExpr));
}